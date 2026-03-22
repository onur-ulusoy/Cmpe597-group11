#!/bin/bash

# 1. Install gdown (best tool for Google Drive)
echo "📦 Installing gdown..."
pip install -U gdown

# 2. Download the file
# ID extracted from your link: 1o1IB6am0HdYS58CEOmmxra3WjJkrn-M1
echo "⬇️  Downloading file from Google Drive..."
gdown 1o1IB6am0HdYS58CEOmmxra3WjJkrn-M1

# 3. Check if it's a zip/tar and offer to extract
echo "✅ Download complete."
echo "ℹ️  If this is a zip or tar file, extract it using:"
echo "   unzip <filename>"
echo "   # or"
echo "   tar -xzvf <filename>"
