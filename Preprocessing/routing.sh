duarouter --trip-files ../Data/traffic/Manhattan_od_0.1.trips.xml \
          --net-file ../Data/Maps/Manhattan.net.xml \
          --output-file ../Data/traffic/Manhattan_od_0.1.rou.xml \
          --ignore-errors \
          --repair \
          --routing-threads 10 \
          --weights.random-factor 0.5
