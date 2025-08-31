echo "Updating package lists"
sudo apt update -y

echo "Installing FFmpeg"
sudo apt install -y ffmpeg

echo "Installing Video4Linux2 utilities"
sudo apt install -y v4l-utils

echo "Verifying installations"
ffmpeg -version && v4l2-ctl --version

echo "Installation complete!"
