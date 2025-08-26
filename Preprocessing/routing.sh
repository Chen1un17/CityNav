duarouter --trip-files ./data/zhouyuping/LLMNavigation/Data/NYC/traffic//data/zhouyuping/LLMNavigation/Data/NYC/ttan_od_0.1.trips.xml \
          --net-file ./data/zhouyuping/LLMNavigation/Data/NYC/Maps//data/zhouyuping/LLMNavigation/Data/NYC/ttan.net.xml \
          --output-file ./data/zhouyuping/LLMNavigation/Data/NYC/traffic//data/zhouyuping/LLMNavigation/Data/NYC/ttan_od_0.1.rou.xml \
          --ignore-errors \
          --repair \
          --routing-threads 10 \
          --weights.random-factor 0.5
