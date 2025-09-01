duarouter --route-files /data/zhouyuping/LLMNavigation/Data/Region_1/Manhattan_od_0.01.rou.alt.xml \
          --net-file /data/zhouyuping/LLMNavigation/Data/Region_1/Manhattan.net.xml \
          --output-file /data/zhouyuping/LLMNavigation/Data/Region_1/Manhattan_od_0.01_processed.rou.xml \
          --ignore-errors \
          --repair \
          --routing-threads 10 \
          --weights.random-factor 0.5
