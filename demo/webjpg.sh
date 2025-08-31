# Set the device path 
DEVICE="/dev/video0"

# Set image resolution
RESOLUTION="640x480"

# Delay before capturing 
DELAY=3

# Output directory and filename
OUTPUT_PATH="test.jpg"

# Wait before capturing
echo "Waiting $DELAY seconds before capturing..."
sleep $DELAY

# Capture the image
ffmpeg -f v4l2 -video_size $RESOLUTION -i $DEVICE -frames:v 1 "$OUTPUT_PATH"

echo "Photo captured and saved as $OUTPUT_PATH"