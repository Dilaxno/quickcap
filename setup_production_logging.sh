#!/bin/bash

# QuickCap Production Logging Setup Script
# This script sets up the necessary directories and permissions for production logging

set -e

echo "🚀 Setting up QuickCap production logging..."

# Create log directories
echo "📁 Creating log directories..."
sudo mkdir -p /var/log/quickcap
sudo mkdir -p /var/run/quickcap

# Create quickcap user if it doesn't exist
if ! id "quickcap" &>/dev/null; then
    echo "👤 Creating quickcap user..."
    sudo useradd --system --home /opt/quickcap --shell /bin/false quickcap
else
    echo "👤 quickcap user already exists"
fi

# Set permissions
echo "🔐 Setting permissions..."
sudo chown quickcap:quickcap /var/log/quickcap
sudo chmod 755 /var/log/quickcap

sudo chown quickcap:quickcap /var/run/quickcap
sudo chmod 755 /var/run/quickcap

# Create logrotate configuration
echo "🔄 Setting up log rotation..."
sudo tee /etc/logrotate.d/quickcap > /dev/null <<EOF
/var/log/quickcap/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 quickcap quickcap
    postrotate
        systemctl reload quickcap.service
    endscript
}
EOF

# Install systemd service
if [ -f "quickcap.service" ]; then
    echo "🔧 Installing systemd service..."
    sudo cp quickcap.service /etc/systemd/system/
    sudo systemctl daemon-reload
    echo "✅ Systemd service installed. Enable with: sudo systemctl enable quickcap.service"
else
    echo "⚠️  quickcap.service file not found. Please copy it manually to /etc/systemd/system/"
fi

echo "✅ Production logging setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy your backend code to /opt/quickcap/backend"
echo "2. Install dependencies in virtual environment"
echo "3. Set environment variables in systemd service or .env file"
echo "4. Enable and start the service:"
echo "   sudo systemctl enable quickcap.service"
echo "   sudo systemctl start quickcap.service"
echo ""
echo "📊 Monitor logs with:"
echo "   tail -f /var/log/quickcap/quickcap.log"
echo "   tail -f /var/log/quickcap/quickcap_error.log"
echo "   journalctl -u quickcap.service -f"