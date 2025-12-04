#!/bin/bash
# Restart Gunicorn with updated code

echo "Stopping Gunicorn..."
killall -9 gunicorn 2>/dev/null
killall -9 python3 2>/dev/null
sleep 2

echo "Starting Gunicorn..."
cd /home/ubuntu/FB-Bot_Python
/usr/bin/python3 -m gunicorn --workers 2 --bind 127.0.0.1:5000 --timeout 120 --daemon app:app

sleep 3

echo "Checking status..."
ps aux | grep gunicorn | grep -v grep

echo ""
echo "Testing API..."
curl -s https://fqbbot.com/api/health | head -5

echo ""
echo "Done!"

