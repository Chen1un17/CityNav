"""
Agent Logger for Multi-Agent Traffic Control System

Provides comprehensive logging and monitoring functionality including:
- Real-time vehicle tracking
- Performance metrics logging
- LLM call monitoring
- Progress display with console output
- Detailed decision logs
"""

import os
import time
import json
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class VehicleStatus:
    """Vehicle status data structure."""
    vehicle_id: str
    current_edge: str
    destination: str
    region: int
    travel_time: float
    last_update: float


@dataclass 
class LLMCallLog:
    """LLM call logging data structure."""
    agent_type: str
    agent_id: str
    call_start: float
    call_end: float
    success: bool
    decision: str
    context_length: int
    error: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    total_vehicles: int
    active_vehicles: int
    completed_vehicles: int
    average_travel_time: float
    system_throughput: float
    regional_metrics: Dict[int, Dict[str, float]]
    
    
class AgentLogger:
    """
    Comprehensive logging system for multi-agent traffic control.
    
    Features:
    - Real-time vehicle tracking and logging
    - Performance metrics collection
    - LLM call monitoring and logging
    - Console progress display
    - Detailed decision and error logging
    """
    
    def __init__(self, log_dir: str = "logs", console_output: bool = True):
        """
        Initialize the logging system.
        
        Args:
            log_dir: Directory for log files
            console_output: Whether to display progress on console
        """
        self.log_dir = log_dir
        self.console_output = console_output
        self.session_start = time.time()
        
        # Step tracking for progress display
        self.current_step = 0
        self.max_steps = 43200  # Default, will be updated
        self.simulation_start_time = 0.0
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log files
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.vehicle_log_file = os.path.join(log_dir, f"vehicles_{self.session_id}.jsonl")
        self.llm_log_file = os.path.join(log_dir, f"llm_calls_{self.session_id}.jsonl")
        self.performance_log_file = os.path.join(log_dir, f"performance_{self.session_id}.jsonl")
        self.system_log_file = os.path.join(log_dir, f"system_{self.session_id}.log")
        
        # Data tracking
        self.vehicle_statuses: Dict[str, VehicleStatus] = {}
        self.llm_calls: List[LLMCallLog] = []
        self.performance_history: List[PerformanceMetrics] = []
        self.travel_times: deque = deque(maxlen=1000)  # Keep last 1000 travel times
        
        # Metrics tracking
        self.total_vehicles = 0
        self.completed_vehicles = 0
        self.active_llm_calls = 0
        self.total_llm_calls = 0
        self.successful_llm_calls = 0
        
        # Threading for logging
        self.log_lock = threading.Lock()
        self.last_console_update = 0
        self.console_update_interval = 2.0  # Update console every 2 seconds for better responsiveness
        
        # Initialize log files
        self._initialize_log_files()
        
        self.log_info("AgentLogger initialized")
    
    def _initialize_log_files(self):
        """Initialize log files with headers."""
        # System log header
        with open(self.system_log_file, 'w') as f:
            f.write(f"Multi-Agent Traffic Control System Log\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log_info(self, message: str):
        """Log an info message."""
        self._log_system_message("INFO", message)
    
    def log_error(self, message: str):
        """Log an error message."""
        self._log_system_message("ERROR", message)
        if self.console_output:
            print(f"ERROR: {message}")
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self._log_system_message("WARNING", message)
    
    def _log_system_message(self, level: str, message: str):
        """Log a system message with timestamp and step information."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add step information if available
        step_info = ""
        if hasattr(self, 'current_step') and self.current_step > 0:
            progress_pct = (self.current_step / self.max_steps * 100) if self.max_steps > 0 else 0
            step_info = f" [Step {self.current_step:.0f}/{self.max_steps:.0f} ({progress_pct:.1f}%)]"
        
        log_line = f"[{timestamp}]{step_info} {level}: {message}\n"
        
        with self.log_lock:
            with open(self.system_log_file, 'a') as f:
                f.write(log_line)
    
    def update_vehicle_count(self, total: int, current_time: float):
        """Update total vehicle count."""
        self.total_vehicles = total
        self.log_info(f"Vehicle count updated: {total} total vehicles")
    
    def update_simulation_params(self, max_steps: int, step_size: float = 1.0):
        """Update simulation parameters for progress tracking."""
        self.max_steps = max_steps
        self.step_size = step_size
        self.log_info(f"Simulation parameters updated: {max_steps} max steps, {step_size} step size")
    
    def log_vehicle_status(self, vehicle_id: str, current_edge: str, destination: str, 
                          region: int, travel_time: float, timestamp: float):
        """Log vehicle status update."""
        status = VehicleStatus(
            vehicle_id=vehicle_id,
            current_edge=current_edge,
            destination=destination,
            region=region,
            travel_time=travel_time,
            last_update=timestamp
        )
        
        with self.log_lock:
            self.vehicle_statuses[vehicle_id] = status
            
            # Write to vehicle log file
            log_entry = {
                "timestamp": timestamp,
                "vehicle_id": vehicle_id,
                "current_edge": current_edge,
                "destination": destination,
                "region": region,
                "travel_time": travel_time,
                "event": "status_update"
            }
            
            with open(self.vehicle_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def log_vehicle_completion(self, vehicle_id: str, start_time: float, 
                             end_time: float, travel_time: float):
        """Log vehicle completion."""
        with self.log_lock:
            self.completed_vehicles += 1
            self.travel_times.append(travel_time)
            
            # Remove from active tracking
            if vehicle_id in self.vehicle_statuses:
                del self.vehicle_statuses[vehicle_id]
            
            # Log completion
            log_entry = {
                "timestamp": end_time,
                "vehicle_id": vehicle_id,
                "start_time": start_time,
                "end_time": end_time,
                "travel_time": travel_time,
                "event": "completion"
            }
            
            with open(self.vehicle_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def log_llm_call_start(self, agent_type: str, agent_id: str, context_length: int, 
                          task_type: str = "decision", input_preview: str = "") -> str:
        """Log the start of an LLM call and return call ID with enhanced tracking."""
        call_id = f"{agent_type}_{agent_id}_{int(time.time() * 1000)}"  # Use milliseconds for uniqueness
        start_time = time.time()
        
        print(f"LLM_CALL_START: {call_id} - {task_type} for {agent_type}({agent_id})")
        print(f"LLM_INPUT_LENGTH: {context_length} characters")
        if input_preview:
            preview = input_preview[:200] + "..." if len(input_preview) > 200 else input_preview
            print(f"LLM_INPUT_PREVIEW: {preview}")
        
        with self.log_lock:
            self.active_llm_calls += 1
            self.total_llm_calls += 1
        
        # Store call metadata for detailed logging
        if not hasattr(self, '_active_call_metadata'):
            self._active_call_metadata = {}
        
        self._active_call_metadata[call_id] = {
            'agent_type': agent_type,
            'agent_id': agent_id,
            'start_time': start_time,
            'task_type': task_type,
            'input_length': context_length,
            'input_preview': input_preview[:500]  # Store first 500 chars
        }
        
        return call_id
    
    def log_llm_call_end(self, call_id: str, success: bool, decision: str, 
                        context_length: int, error: Optional[str] = None):
        """Log the completion of an LLM call with detailed output tracking."""
        parts = call_id.split('_')
        agent_type = parts[0] if len(parts) >= 1 else "unknown"
        agent_id = parts[1] if len(parts) >= 2 else "unknown"
        
        end_time = time.time()
        
        # Get start time from stored metadata (more reliable than parsing call_id)
        call_metadata = getattr(self, '_active_call_metadata', {}).get(call_id, {})
        start_time = call_metadata.get('start_time', end_time - 1)
        
        # Calculate duration in milliseconds
        duration_ms = (end_time - start_time) * 1000
        
        # Get additional metadata
        task_type = call_metadata.get('task_type', 'decision')
        input_preview = call_metadata.get('input_preview', '')
        
        # Console output for detailed tracking
        status_text = "SUCCESS" if success else "FAILED"
        print(f"LLM_CALL_END: {call_id} - {status_text} ({duration_ms:.1f}ms)")
        
        if success:
            # Show decision preview
            decision_preview = decision[:150] + "..." if len(decision) > 150 else decision
            print(f"LLM_DECISION: {decision_preview}")
        else:
            print(f"LLM_ERROR: {error}")
        
        if input_preview:
            print(f"LLM_INPUT_SUMMARY: {input_preview[:100]}...")
            
        # Performance warnings
        if duration_ms > 10000:  # > 10 seconds
            print(f"LLM_PERFORMANCE_WARNING: Call took {duration_ms:.1f}ms")
        elif duration_ms > 30000:  # > 30 seconds
            print(f"LLM_PERFORMANCE_CRITICAL: Call took {duration_ms:.1f}ms - check model/network")
            
        call_log = LLMCallLog(
            agent_type=agent_type,
            agent_id=agent_id,
            call_start=start_time,
            call_end=end_time,
            success=success,
            decision=decision,
            context_length=context_length,
            error=error
        )
        
        with self.log_lock:
            self.active_llm_calls -= 1
            if success:
                self.successful_llm_calls += 1
            
            self.llm_calls.append(call_log)
            
            # Enhanced log entry with more details
            log_entry = {
                "timestamp": end_time,
                "agent_type": agent_type,
                "agent_id": agent_id,
                "task_type": task_type,
                "duration_ms": duration_ms,
                "success": success,
                "decision": decision,
                "decision_length": len(decision) if decision else 0,
                "context_length": context_length,
                "input_preview": input_preview[:200] if input_preview else "",
                "error": error,
                "performance_category": self._categorize_performance(duration_ms)
            }
            
            with open(self.llm_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        # Clean up call metadata
        if hasattr(self, '_active_call_metadata') and call_id in self._active_call_metadata:
            del self._active_call_metadata[call_id]
    
    def _categorize_performance(self, duration_ms: float) -> str:
        """Categorize LLM call performance based on duration."""
        if duration_ms < 2000:  # < 2 seconds
            return "excellent"
        elif duration_ms < 5000:  # < 5 seconds
            return "good"
        elif duration_ms < 10000:  # < 10 seconds
            return "acceptable"
        elif duration_ms < 30000:  # < 30 seconds
            return "slow"
        else:
            return "critical"
    
    def log_system_performance(self, regional_metrics: Dict[int, Dict], 
                             traffic_metrics: Dict, prediction_metrics: Dict, 
                             timestamp: float):
        """Log comprehensive system performance metrics."""
        active_vehicles = len(self.vehicle_statuses)
        avg_travel_time = sum(self.travel_times) / len(self.travel_times) if self.travel_times else 0.0
        throughput = self.completed_vehicles / (timestamp - self.session_start) if timestamp > self.session_start else 0.0
        
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            total_vehicles=self.total_vehicles,
            active_vehicles=active_vehicles,
            completed_vehicles=self.completed_vehicles,
            average_travel_time=avg_travel_time,
            system_throughput=throughput,
            regional_metrics=regional_metrics
        )
        
        with self.log_lock:
            self.performance_history.append(metrics)
            
            # Write to performance log file
            log_entry = {
                "timestamp": timestamp,
                "total_vehicles": self.total_vehicles,
                "active_vehicles": active_vehicles,
                "completed_vehicles": self.completed_vehicles,
                "average_travel_time": avg_travel_time,
                "system_throughput": throughput,
                "regional_metrics": regional_metrics,
                "traffic_metrics": traffic_metrics,
                "prediction_metrics": prediction_metrics,
                "llm_stats": {
                    "total_calls": self.total_llm_calls,
                    "successful_calls": self.successful_llm_calls,
                    "active_calls": self.active_llm_calls,
                    "success_rate": self.successful_llm_calls / self.total_llm_calls if self.total_llm_calls > 0 else 0.0
                }
            }
            
            with open(self.performance_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def display_progress(self, current_time: float, current_step: float = None):
        """Display progress information on console with step-based progress."""
        if not self.console_output:
            return
            
        # Update step information
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step = current_time  # Fallback to using time as step
            
        if current_time - self.last_console_update < self.console_update_interval:
            return
            
        self.last_console_update = current_time
        
        # Calculate step-based progress
        step_progress = self.current_step / self.max_steps if self.max_steps > 0 else 0.0
        step_progress_pct = step_progress * 100
        
        # Calculate time metrics
        wall_clock_elapsed = time.time() - self.session_start
        simulation_time = current_time
        
        # Calculate vehicle metrics
        active_vehicles = len(self.vehicle_statuses)
        avg_travel_time = sum(self.travel_times) / len(self.travel_times) if self.travel_times else 0.0
        throughput = self.completed_vehicles / wall_clock_elapsed if wall_clock_elapsed > 0 else 0.0
        llm_success_rate = self.successful_llm_calls / self.total_llm_calls if self.total_llm_calls > 0 else 0.0
        
        # Create step-based progress bar
        progress_bar = self._create_progress_bar(self.current_step, self.max_steps)
        
        # Enhanced console display with step information
        print(f"\r{progress_bar} | "
              f"Step: {self.current_step:.0f}/{self.max_steps:.0f} ({step_progress_pct:.1f}%) | "
              f"SimTime: {simulation_time:.1f}s | "
              f"WallTime: {wall_clock_elapsed/60:.1f}m | "
              f"Vehicles: {active_vehicles}/{self.total_vehicles} | "
              f"Completed: {self.completed_vehicles} | "
              f"ATT: {avg_travel_time:.1f}s | "
              f"Rate: {throughput:.2f}/s | "
              f"LLM: {llm_success_rate:.0%}", 
              end='', flush=True)
        
        # Periodic detailed log output
        if int(self.current_step) % 100 == 0:  # Every 100 steps
            self.log_info(f"PROGRESS: Step {self.current_step:.0f}/{self.max_steps:.0f} ({step_progress_pct:.1f}%) | "
                         f"SimTime: {simulation_time:.1f}s | Active: {active_vehicles} | "
                         f"Completed: {self.completed_vehicles} | ATT: {avg_travel_time:.1f}s")
    
    def _create_progress_bar(self, current: float, total: float, width: int = 30) -> str:
        """Create a text progress bar."""
        if total <= 0:
            return "[" + "?" * width + "]"
        
        progress = min(current / total, 1.0)
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {progress:.1%}"
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        active_vehicles = len(self.vehicle_statuses)
        avg_travel_time = sum(self.travel_times) / len(self.travel_times) if self.travel_times else 0.0
        current_time = time.time()
        throughput = self.completed_vehicles / (current_time - self.session_start) if current_time > self.session_start else 0.0
        llm_success_rate = self.successful_llm_calls / self.total_llm_calls if self.total_llm_calls > 0 else 0.0
        
        return {
            "total_vehicles": self.total_vehicles,
            "active_vehicles": active_vehicles,
            "completed_vehicles": self.completed_vehicles,
            "average_travel_time": avg_travel_time,
            "system_throughput": throughput,
            "llm_stats": {
                "total_calls": self.total_llm_calls,
                "successful_calls": self.successful_llm_calls,
                "active_calls": self.active_llm_calls,
                "success_rate": llm_success_rate
            },
            "session_duration": current_time - self.session_start
        }
    
    def close_session(self):
        """Close the logging session and generate summary."""
        session_end = time.time()
        session_duration = session_end - self.session_start
        
        # Generate final summary
        summary = self.get_performance_summary()
        summary["session_end"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        summary["session_duration_hours"] = session_duration / 3600
        
        # Write summary to system log
        self.log_info("="*50)
        self.log_info("SESSION SUMMARY")
        self.log_info("="*50)
        self.log_info(f"Duration: {session_duration/3600:.2f} hours")
        self.log_info(f"Total vehicles: {summary['total_vehicles']}")
        self.log_info(f"Completed vehicles: {summary['completed_vehicles']}")
        self.log_info(f"Average travel time: {summary['average_travel_time']:.1f}s")
        self.log_info(f"System throughput: {summary['system_throughput']:.2f} vehicles/s")
        self.log_info(f"LLM calls: {summary['llm_stats']['total_calls']} total, "
                     f"{summary['llm_stats']['success_rate']:.1%} success rate")
        
        # Write summary file
        summary_file = os.path.join(self.log_dir, f"summary_{self.session_id}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.console_output:
            print(f"\n\nSession completed. Logs saved to: {self.log_dir}")
            print(f"Summary: {summary['completed_vehicles']}/{summary['total_vehicles']} vehicles completed")
            print(f"Average travel time: {summary['average_travel_time']:.1f}s")
            print(f"System throughput: {summary['system_throughput']:.2f} vehicles/s")