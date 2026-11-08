#!/bin/sh
set -eu

cd /app
mkdir -p /data /app/results

if [ ! -e /app/calibration.json ]; then
    if [ ! -f /data/calibration.json ]; then
        : > /data/calibration.json
    fi
    ln -s /data/calibration.json /app/calibration.json
fi

if [ "$#" -gt 0 ]; then
    case "$1" in
        python|bash|sh)
            exec "$@"
            ;;
    esac
fi

if [ "${SHOTTRACKER_NOVNC:-1}" = "1" ]; then
    export DISPLAY="${SHOTTRACKER_DISPLAY:-:99}"
    screen=${DISPLAY#:}
    # После перезапуска контейнера lock и сокет остаются, а процесс Xvfb уже мёртв.
    rm -f "/tmp/.X${screen}-lock" "/tmp/.X11-unix/X${screen}"
    Xvfb "$DISPLAY" -screen 0 1400x900x24 -ac +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
    i=0
    while [ ! -S "/tmp/.X11-unix/X${screen}" ]; do
        i=$((i + 1))
        if [ "$i" -gt 50 ]; then
            echo "Xvfb не поднялся" >&2
            cat /tmp/xvfb.log >&2 || true
            exit 1
        fi
        sleep 0.1
    done
    x11vnc -display "$DISPLAY" -nopw -forever -shared -wait 5 -defer 5 -rfbport 5900 >/tmp/x11vnc.log 2>&1 &
    websockify --web=/usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
    echo "Окно ShotTracker: http://localhost:6080/vnc.html?autoconnect=1&resize=scale"
fi

exec python app.py "$@"
