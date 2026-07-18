# Oracle Backend Deployment

This deploys only the RCM API backend to an Ubuntu Oracle Cloud VM.

```text
http://SERVER_IP/api/config
https://rcm-api.michaeljirasek.com/api/config
```

## 1. Create the VM

In Oracle Cloud, create an Always Free compute instance:

- Image: Ubuntu 24.04 or 22.04.
- Shape: Always Free eligible.
- Networking: assign a public IPv4 address.
- SSH: paste the public key from this PC.

Add ingress rules for TCP ports 80 and 443 in the Oracle subnet security list or network security group. Keep TCP port 22 open only for your IP if possible.

## 2. Upload From Windows

From PowerShell on this PC:

```powershell
cd C:\MJ_Python\matlab_oxford\rcm-python
PowerShell -ExecutionPolicy Bypass -File deploy\oracle\package_windows.ps1
scp -i C:\Users\group\.ssh\oracle_rcm_ed25519 C:\MJ_Python\matlab_oxford\rcm-python.tar.gz ubuntu@SERVER_IP:/tmp/
```

On the Oracle VM:

```bash
sudo mkdir -p /opt/rcm-python
sudo tar -xzf /tmp/rcm-python.tar.gz -C /opt/rcm-python
sudo chown -R ubuntu:ubuntu /opt/rcm-python
cd /opt/rcm-python
bash deploy/oracle/install_ubuntu.sh
```

## 3. DNS and HTTPS

Create this DNS record at INWX:

```text
rcm-api.michaeljirasek.com  A  SERVER_IP
```

After DNS resolves to the Oracle VM:

```bash
cd /opt/rcm-python
EMAIL=you@example.com bash deploy/oracle/enable_https.sh
```

## 4. Server Commands

```bash
sudo systemctl status rcm-api
sudo journalctl -u rcm-api -n 100 --no-pager
sudo systemctl restart rcm-api
```
