import asyncio
import time
import threading
import queue
import logging
import json
import traceback
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Callable, List
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from collections import defaultdict, deque
import weakref

@dataclass
class RequestMetrics:
    """Track metrics for request performance monitoring"""
    request_id: str
    timestamp: float
    duration: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    retry_count: int = 0
    queue_time: float = 0.0

class AdaptiveBackoff:
    """Adaptive backoff strategy for request retries"""
    def __init__(self, base_delay=1.0, max_delay=60.0, factor=1.5):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.factor = factor
        self.failure_counts = {}
        self.success_counts = {}
        
    def get_delay(self, key: str) -> float:
        """Get backoff delay for a specific key (e.g., agent_id)"""
        failures = self.failure_counts.get(key, 0)
        delay = min(self.base_delay * (self.factor ** failures), self.max_delay)
        return delay
    
    def record_failure(self, key: str):
        """Record a failure for backoff calculation"""
        self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
        
    def record_success(self, key: str):
        """Record a success, reduce backoff"""
        if key in self.failure_counts:
            self.failure_counts[key] = max(0, self.failure_counts[key] - 1)
        self.success_counts[key] = self.success_counts.get(key, 0) + 1

class ConcurrencyManager:
    """
    Advanced concurrency manager for vLLM requests with intelligent queuing,
    backpressure, and adaptive rate limiting.
    """
    
    def __init__(self, 
                 max_concurrent_requests=8,
                 max_queue_size=100,
                 request_timeout=30.0,
                 rate_limit_per_second=10.0,
                 adaptive_scaling=True):
        """
        Initialize concurrency manager
        
        Args:
            max_concurrent_requests: Maximum simultaneous requests to vLLM
            max_queue_size: Maximum queued requests before rejection
            request_timeout: Request timeout in seconds
            rate_limit_per_second: Maximum requests per second
            adaptive_scaling: Enable adaptive rate limiting based on performance
        """
        self.max_concurrent_requests = max_concurrent_requests
        self.max_queue_size = max_queue_size
        self.request_timeout = request_timeout
        self.rate_limit_per_second = rate_limit_per_second
        self.adaptive_scaling = adaptive_scaling
        
        # Concurrency control
        self.semaphore = threading.Semaphore(max_concurrent_requests)
        self.request_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.active_requests = set()  # Changed from WeakSet to regular set
        
        # Rate limiting
        self.last_request_times = []
        self.rate_limit_lock = threading.Lock()
        
        # Metrics and monitoring
        self.metrics = {}
        self.metrics_lock = threading.Lock()
        self.backoff = AdaptiveBackoff()
        
        # Circuit breaker state
        self.circuit_breaker = {
            'state': 'CLOSED',  # CLOSED, OPEN, HALF_OPEN
            'failure_count': 0,
            'last_failure_time': 0,
            'failure_threshold': 5,
            'timeout': 30.0
        }
        
        # Performance tracking and monitoring
        self.performance_window = []
        self.performance_lock = threading.Lock()
        self.call_records = deque(maxlen=500)  # Keep last 500 calls
        self.agent_stats = defaultdict(lambda: {
            'total_calls': 0, 'successful_calls': 0, 'failed_calls': 0,
            'total_duration': 0, 'avg_duration': 0, 'max_duration': 0,
            'min_duration': float('inf'), 'empty_responses': 0,
            'error_types': defaultdict(int), 'last_call_time': 0
        })
        
        # Alert thresholds
        self.alert_thresholds = {
            'max_duration_ms': 25000, 'failure_rate_threshold': 0.25,
            'empty_response_rate_threshold': 0.15, 'consecutive_failures_threshold': 4
        }
        
        # Logging setup
        self.logger = logging.getLogger(f"ConcurrencyManager-{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        # Setup monitoring log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.monitor_log_file = f"llm_monitoring_{timestamp}.jsonl"
        
        print(f"ConcurrencyManager initialized:")
        print(f"  - Max concurrent: {max_concurrent_requests}")
        print(f"  - Queue size: {max_queue_size}")
        print(f"  - Rate limit: {rate_limit_per_second}/s")
        print(f"  - Request timeout: {request_timeout}s")
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows requests"""
        current_time = time.time()
        
        if self.circuit_breaker['state'] == 'OPEN':
            if current_time - self.circuit_breaker['last_failure_time'] > self.circuit_breaker['timeout']:
                self.circuit_breaker['state'] = 'HALF_OPEN'
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                return False
        
        return True
    
    def _record_request_result(self, success: bool, agent_id: str):
        """Record request result for circuit breaker and adaptive scaling"""
        if success:
            self.circuit_breaker['failure_count'] = max(0, self.circuit_breaker['failure_count'] - 1)
            if self.circuit_breaker['state'] == 'HALF_OPEN':
                self.circuit_breaker['state'] = 'CLOSED'
                self.logger.info("Circuit breaker reset to CLOSED")
            self.backoff.record_success(agent_id)
        else:
            self.circuit_breaker['failure_count'] += 1
            self.circuit_breaker['last_failure_time'] = time.time()
            if self.circuit_breaker['failure_count'] >= self.circuit_breaker['failure_threshold']:
                self.circuit_breaker['state'] = 'OPEN'
                self.logger.warning(f"Circuit breaker OPEN due to {self.circuit_breaker['failure_count']} failures")
            self.backoff.record_failure(agent_id)
    
    def _enforce_rate_limit(self) -> bool:
        """Enforce rate limiting with sliding window"""
        current_time = time.time()
        
        with self.rate_limit_lock:
            # Remove old entries (older than 1 second)
            self.last_request_times = [t for t in self.last_request_times if current_time - t < 1.0]
            
            # Check if we can make a new request
            if len(self.last_request_times) >= self.rate_limit_per_second:
                # Calculate wait time until oldest request expires
                oldest_time = min(self.last_request_times)
                wait_time = 1.0 - (current_time - oldest_time)
                if wait_time > 0:
                    time.sleep(wait_time)
                    current_time = time.time()
                    self.last_request_times = [t for t in self.last_request_times if current_time - t < 1.0]
            
            # Record this request
            self.last_request_times.append(current_time)
            return True
    
    def _adaptive_rate_adjustment(self):
        """Adjust rate limits based on recent performance"""
        if not self.adaptive_scaling:
            return
            
        with self.performance_lock:
            if len(self.performance_window) < 10:
                return
                
            # Calculate recent success rate
            recent_success_rate = sum(1 for p in self.performance_window[-10:] if p['success']) / 10
            
            # Adjust rate limit based on success rate
            if recent_success_rate > 0.9 and self.rate_limit_per_second < 20:
                self.rate_limit_per_second = min(20, self.rate_limit_per_second * 1.1)
                self.logger.info(f"Increased rate limit to {self.rate_limit_per_second:.1f}/s")
            elif recent_success_rate < 0.7 and self.rate_limit_per_second > 2:
                self.rate_limit_per_second = max(2, self.rate_limit_per_second * 0.8)
                self.logger.warning(f"Decreased rate limit to {self.rate_limit_per_second:.1f}/s")
    
    @contextmanager
    def managed_request(self, agent_id: str, priority: int = 0):
        """
        Context manager for managed LLM requests
        
        Args:
            agent_id: Identifier for the requesting agent
            priority: Request priority (lower = higher priority)
            
        Usage:
            with concurrency_manager.managed_request("agent_1") as request_id:
                # Make LLM call here
                result = llm.inference(prompt)
        """
        if not self._check_circuit_breaker():
            raise Exception("Circuit breaker is OPEN - service temporarily unavailable")
        
        # Enforce rate limiting
        self._enforce_rate_limit()
        
        # Wait for available slot
        acquired = self.semaphore.acquire(timeout=self.request_timeout)
        if not acquired:
            raise Exception(f"Request timeout: no slot available within {self.request_timeout}s")
        
        request_id = f"{agent_id}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Create metrics tracking
        metrics = RequestMetrics(
            request_id=request_id,
            timestamp=start_time
        )
        
        try:
            # Add to active requests tracking
            self.active_requests.add(request_id)
            
            # Wait for backoff if needed
            backoff_delay = self.backoff.get_delay(agent_id)
            if backoff_delay > 0:
                self.logger.info(f"Applying backoff delay {backoff_delay:.2f}s for {agent_id}")
                time.sleep(backoff_delay)
            
            yield request_id
            
            # Record success
            metrics.success = True
            metrics.duration = time.time() - start_time
            self._record_request_result(True, agent_id)
            
        except Exception as e:
            # Record failure
            metrics.success = False
            metrics.error = str(e)
            metrics.duration = time.time() - start_time
            self._record_request_result(False, agent_id)
            raise
            
        finally:
            # Remove from active requests and release semaphore
            self.active_requests.discard(request_id)
            self.semaphore.release()
            
            with self.metrics_lock:
                self.metrics[request_id] = metrics
                
            # Update performance window
            with self.performance_lock:
                self.performance_window.append({
                    'timestamp': start_time,
                    'duration': metrics.duration,
                    'success': metrics.success,
                    'agent_id': agent_id
                })
                # Keep only last 100 entries
                if len(self.performance_window) > 100:
                    self.performance_window = self.performance_window[-100:]
            
            # Adaptive rate adjustment
            self._adaptive_rate_adjustment()
            
            # Record detailed monitoring data
            self._record_detailed_monitoring(request_id, start_time, metrics)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current concurrency manager status"""
        with self.metrics_lock:
            total_requests = len(self.metrics)
            successful_requests = sum(1 for m in self.metrics.values() if m.success)
            
        current_active = len(self.active_requests)
        available_slots = self.semaphore._value
        
        with self.performance_lock:
            if self.performance_window:
                avg_duration = sum(p['duration'] for p in self.performance_window if p['duration']) / len(self.performance_window)
                recent_success_rate = sum(1 for p in self.performance_window[-20:] if p['success']) / min(20, len(self.performance_window))
            else:
                avg_duration = 0
                recent_success_rate = 1.0
        
        return {
            'circuit_breaker_state': self.circuit_breaker['state'],
            'current_rate_limit': self.rate_limit_per_second,
            'active_requests': current_active,
            'available_slots': available_slots,
            'queue_size': self.request_queue.qsize(),
            'total_requests': total_requests,
            'success_rate': successful_requests / max(1, total_requests),
            'recent_success_rate': recent_success_rate,
            'avg_duration': avg_duration,
            'failure_count': self.circuit_breaker['failure_count']
        }
    
    def print_status(self):
        """Print current status in readable format"""
        status = self.get_status()
        print("\n=== Concurrency Manager Status ===")
        print(f"Circuit Breaker: {status['circuit_breaker_state']}")
        print(f"Rate Limit: {status['current_rate_limit']:.1f}/s")
        print(f"Active Requests: {status['active_requests']}/{self.max_concurrent_requests}")
        print(f"Available Slots: {status['available_slots']}")
        print(f"Queue Size: {status['queue_size']}/{self.max_queue_size}")
        print(f"Success Rate: {status['success_rate']:.2%} (Recent: {status['recent_success_rate']:.2%})")
        print(f"Avg Duration: {status['avg_duration']:.2f}s")
        print(f"Failures: {status['failure_count']}")
        print("=" * 35)
    
    def _record_detailed_monitoring(self, request_id: str, start_time: float, metrics: RequestMetrics):
        """Record detailed monitoring data for performance analysis"""
        try:
            # Extract agent_id from request_id
            agent_id = request_id.split('_')[0]
            
            # Update agent statistics
            with self.metrics_lock:
                stats = self.agent_stats[agent_id]
                stats['total_calls'] += 1
                stats['last_call_time'] = start_time
                
                if metrics.success:
                    stats['successful_calls'] += 1
                    if metrics.duration:
                        stats['total_duration'] += metrics.duration
                        stats['avg_duration'] = stats['total_duration'] / stats['successful_calls']
                        stats['max_duration'] = max(stats['max_duration'], metrics.duration)
                        stats['min_duration'] = min(stats['min_duration'], metrics.duration)
                else:
                    stats['failed_calls'] += 1
                    if metrics.error:
                        error_type = type(Exception(metrics.error)).__name__
                        stats['error_types'][error_type] += 1
                
                # Track empty responses specifically
                if metrics.success and metrics.duration and metrics.duration < 0.1:
                    stats['empty_responses'] += 1
                
                # Add to call records for detailed analysis
                self.call_records.append({
                    'request_id': request_id,
                    'agent_id': agent_id,
                    'timestamp': start_time,
                    'duration': metrics.duration,
                    'success': metrics.success,
                    'error': metrics.error,
                    'retry_count': metrics.retry_count
                })
                
                # Check for alerts
                self._check_performance_alerts(agent_id, stats)
                
        except Exception as e:
            print(f"Error recording monitoring data: {e}")
    
    def _check_performance_alerts(self, agent_id: str, stats: Dict):
        """Check for performance issues and generate alerts"""
        try:
            total_calls = stats['total_calls']
            if total_calls < 5:  # Not enough data for reliable alerts
                return
            
            # Check failure rate
            failure_rate = stats['failed_calls'] / total_calls
            if failure_rate > self.alert_thresholds['failure_rate_threshold']:
                print(f"ALERT: High failure rate for {agent_id}: {failure_rate:.2%}")
            
            # Check empty response rate
            empty_rate = stats['empty_responses'] / total_calls
            if empty_rate > self.alert_thresholds['empty_response_rate_threshold']:
                print(f"ALERT: High empty response rate for {agent_id}: {empty_rate:.2%}")
            
            # Check average duration
            if stats['avg_duration'] > self.alert_thresholds['max_duration_ms'] / 1000:
                print(f"ALERT: High average duration for {agent_id}: {stats['avg_duration']:.2f}s")
                
        except Exception as e:
            print(f"Error checking performance alerts: {e}")
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        with self.metrics_lock:
            report = {
                'timestamp': time.time(),
                'system_status': self.get_status(),
                'agent_statistics': dict(self.agent_stats),
                'recent_calls': list(self.call_records)[-50:]  # Last 50 calls
            }
            
            # Calculate system-wide metrics
            total_calls = sum(stats['total_calls'] for stats in self.agent_stats.values())
            total_failures = sum(stats['failed_calls'] for stats in self.agent_stats.values())
            total_empty = sum(stats['empty_responses'] for stats in self.agent_stats.values())
            
            report['system_summary'] = {
                'total_calls': total_calls,
                'system_failure_rate': total_failures / max(1, total_calls),
                'system_empty_rate': total_empty / max(1, total_calls),
                'active_agents': len(self.agent_stats)
            }
            
            return report

class EnhancedConnectionPool:
    """Enhanced connection pool with better concurrency control"""
    
    def __init__(self, base_url, pool_size=3, max_retries=3, timeout=30, concurrency_manager=None):
        self.base_url = base_url
        self.pool_size = pool_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.concurrency_manager = concurrency_manager
        
        # Connection management
        self._pool = queue.Queue(maxsize=pool_size)
        self._lock = threading.RLock()
        self._created_clients = 0
        self._healthy_clients = set()
        self._client_usage_count = {}
        
        # Health monitoring
        self._last_health_check = 0
        self._health_check_interval = 60  # seconds
        
        # Initialize pool
        self._initialize_pool()
        
        # Start background health monitor
        self._start_health_monitor()
    
    def _initialize_pool(self):
        """Initialize connection pool with health checks"""
        for i in range(self.pool_size):
            try:
                client = self._create_client()
                if client and self._test_client_health(client):
                    self._pool.put(client)
                    self._healthy_clients.add(id(client))
                    self._client_usage_count[id(client)] = 0
            except Exception as e:
                print(f"Failed to create initial client {i}: {e}")
    
    def _create_client(self):
        """Create a new OpenAI client with optimized settings"""
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=self.base_url, 
                api_key="EMPTY", 
                timeout=self.timeout,
                max_retries=0  # Handle retries at our level
            )
            self._created_clients += 1
            return client
        except Exception as e:
            print(f"Error creating client: {e}")
            return None
    
    def _test_client_health(self, client) -> bool:
        """Test if client is healthy"""
        try:
            # Quick health check
            response = client.models.list()
            return response is not None
        except Exception:
            return False
    
    def _start_health_monitor(self):
        """Start background health monitoring"""
        def health_monitor():
            while True:
                try:
                    time.sleep(self._health_check_interval)
                    self._periodic_health_check()
                except Exception as e:
                    print(f"Health monitor error: {e}")
        
        monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        monitor_thread.start()
    
    def _periodic_health_check(self):
        """Periodic health check of all clients"""
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return
        
        self._last_health_check = current_time
        
        with self._lock:
            # Collect all clients for testing
            clients_to_test = []
            while not self._pool.empty():
                try:
                    client = self._pool.get_nowait()
                    clients_to_test.append(client)
                except queue.Empty:
                    break
            
            # Test and filter healthy clients
            healthy_clients = []
            for client in clients_to_test:
                if self._test_client_health(client):
                    healthy_clients.append(client)
                    self._healthy_clients.add(id(client))
                else:
                    self._healthy_clients.discard(id(client))
                    if id(client) in self._client_usage_count:
                        del self._client_usage_count[id(client)]
            
            # Return healthy clients to pool
            for client in healthy_clients:
                try:
                    self._pool.put_nowait(client)
                except queue.Full:
                    break
            
            # Create new clients if pool is undersized
            current_size = len(healthy_clients)
            for i in range(self.pool_size - current_size):
                try:
                    new_client = self._create_client()
                    if new_client and self._test_client_health(new_client):
                        self._pool.put_nowait(new_client)
                        self._healthy_clients.add(id(new_client))
                        self._client_usage_count[id(new_client)] = 0
                except queue.Full:
                    break
            
            print(f"Health check completed: {len(healthy_clients)} healthy clients")
    
    @contextmanager
    def get_client(self, agent_id: str = "unknown"):
        """Get client with enhanced error handling and monitoring"""
        client = None
        start_time = time.time()
        
        try:
            # Use concurrency manager if available
            if self.concurrency_manager:
                with self.concurrency_manager.managed_request(agent_id) as request_id:
                    client = self._get_client_from_pool(agent_id)
                    if client:
                        yield client
                        # Record usage
                        self._client_usage_count[id(client)] = self._client_usage_count.get(id(client), 0) + 1
                    else:
                        raise Exception("No healthy client available")
            else:
                # Fallback without concurrency manager
                client = self._get_client_from_pool(agent_id)
                if client:
                    yield client
                    self._client_usage_count[id(client)] = self._client_usage_count.get(id(client), 0) + 1
                else:
                    raise Exception("No healthy client available")
                
        except Exception as e:
            # Mark client as unhealthy if it caused the error
            if client and id(client) in self._healthy_clients:
                self._healthy_clients.discard(id(client))
                print(f"Marked client {id(client)} as unhealthy due to error: {e}")
            raise
        finally:
            # Return client to pool if still healthy
            if client and id(client) in self._healthy_clients:
                try:
                    self._pool.put_nowait(client)
                except queue.Full:
                    # Pool is full, client will be garbage collected
                    pass
    
    def _get_client_from_pool(self, agent_id: str):
        """Get client from pool or create new one"""
        try:
            # Try to get existing client
            client = self._pool.get(timeout=2.0)
            if id(client) in self._healthy_clients:
                return client
            else:
                # Client is unhealthy, try to create new one
                return self._create_client()
        except queue.Empty:
            # Pool is empty, create new client
            new_client = self._create_client()
            if new_client and self._test_client_health(new_client):
                self._healthy_clients.add(id(new_client))
                self._client_usage_count[id(new_client)] = 0
                return new_client
            else:
                return None
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status"""
        with self._lock:
            healthy_count = len(self._healthy_clients)
            total_usage = sum(self._client_usage_count.values())
            
            return {
                'pool_size': self.pool_size,
                'healthy_clients': healthy_count,
                'queue_size': self._pool.qsize(),
                'total_clients_created': self._created_clients,
                'total_requests_served': total_usage,
                'avg_requests_per_client': total_usage / max(1, healthy_count)
            }

def create_enhanced_llm_wrapper(original_llm, agent_id: str, concurrency_manager: ConcurrencyManager):
    """
    Create an enhanced wrapper around the original LLM with concurrency control
    """
    
    class EnhancedLLMWrapper:
        def __init__(self, original_llm, agent_id, concurrency_manager):
            self.original_llm = original_llm
            self.agent_id = agent_id
            self.concurrency_manager = concurrency_manager
            
            # Wrap connection pool if it exists
            if hasattr(original_llm, 'connection_pool') and original_llm.connection_pool:
                original_llm.connection_pool = EnhancedConnectionPool(
                    original_llm.connection_pool.base_url,
                    pool_size=3,
                    timeout=original_llm.connection_pool.timeout,
                    concurrency_manager=concurrency_manager
                )
        
        def __getattr__(self, name):
            """Delegate all other attributes to original LLM"""
            return getattr(self.original_llm, name)
        
        def inference(self, query, system_prompt=None):
            """Enhanced inference with concurrency control"""
            try:
                with self.concurrency_manager.managed_request(self.agent_id) as request_id:
                    result = self.original_llm.inference(query, system_prompt)
                    
                    # Validate result
                    if result is None or (isinstance(result, str) and not result.strip()):
                        raise Exception("Empty or None response from LLM")
                    
                    return result
                    
            except Exception as e:
                print(f"Enhanced LLM inference error for {self.agent_id}: {e}")
                # Return a default response instead of None
                return '{"answer": "1", "summary": "Error in LLM inference, using fallback"}'
        
        def batch_inference(self, queries, system_prompt=None):
            """Enhanced batch inference with controlled concurrency"""
            if not queries:
                return []
            
            results = []
            
            # Process in smaller batches to avoid overwhelming the server
            optimal_batch_size = min(4, len(queries))  # Reduced from original batch_size
            
            for i in range(0, len(queries), optimal_batch_size):
                batch_queries = queries[i:i + optimal_batch_size]
                batch_results = []
                
                try:
                    with self.concurrency_manager.managed_request(f"{self.agent_id}_batch") as request_id:
                        # Use original batch_inference but with limited concurrency
                        batch_results = self.original_llm.batch_inference(batch_queries, system_prompt)
                        
                        # Validate each result in batch
                        validated_results = []
                        for j, result in enumerate(batch_results):
                            if result is None or (isinstance(result, str) and not result.strip()):
                                validated_results.append('{"answer": "1", "summary": "Empty response, using fallback"}')
                            else:
                                validated_results.append(result)
                        
                        results.extend(validated_results)
                        
                except Exception as e:
                    print(f"Batch inference error for {self.agent_id}: {e}")
                    # Provide fallback results for the entire batch
                    fallback_results = ['{"answer": "1", "summary": "Batch inference failed, using fallback"}'] * len(batch_queries)
                    results.extend(fallback_results)
                
                # Small delay between batches to prevent overwhelming
                if i + optimal_batch_size < len(queries):
                    time.sleep(0.5)
            
            return results
    
    return EnhancedLLMWrapper(original_llm, agent_id, concurrency_manager)

# Global concurrency manager instance
_global_concurrency_manager = None

def get_global_concurrency_manager() -> ConcurrencyManager:
    """Get or create global concurrency manager"""
    global _global_concurrency_manager
    if _global_concurrency_manager is None:
        _global_concurrency_manager = ConcurrencyManager(
            max_concurrent_requests=6,  # Conservative limit based on vLLM max_num_seqs
            max_queue_size=50,
            request_timeout=45.0,
            rate_limit_per_second=8.0,  # Conservative rate limit
            adaptive_scaling=True
        )
    return _global_concurrency_manager

def reset_global_concurrency_manager():
    """Reset global concurrency manager (useful for testing)"""
    global _global_concurrency_manager
    _global_concurrency_manager = None