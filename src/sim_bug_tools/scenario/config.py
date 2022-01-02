import os
dir_path = os.path.dirname(os.path.realpath(__file__))

SUMO = {

    "gui" : False,

    # Street network
    "--net-file" : "%s/map/merge.net.xml" % dir_path,

    # Logging
    "--error-log" : "%s/log/error-log.txt" % dir_path,

    # Smooth lane changing
    "--lanechange.duration": 2,

    # Split lanes
    # "--lateral-resolution" : "5.5",

    # Traci Connection
    "--num-clients" : 1,
    "--remote-port" : 5522,

    # GUI Options
    "--delay" : 100,
    "--start" : "--quit-on-end",

    # RNG
    "--seed" : 333
}