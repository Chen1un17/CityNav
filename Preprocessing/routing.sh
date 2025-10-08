duarouter --route-files /data/XXXXX/LLMNavigation/Data/Region_1/Manhattan_od_0.01.rou.alt.xml \
          --net-file /data/XXXXX/LLMNavigation/Data/Region_1/Manhattan.net.xml \
          --output-file /data/XXXXX/LLMNavigation/Data/Region_1/Manhattan_od_0.01_processed.rou.xml \
          --ignore-errors \
          --repair \
          --routing-threads 10 \
          --weights.random-factor 0.5
