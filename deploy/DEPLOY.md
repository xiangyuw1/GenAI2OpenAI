# 部署指南

## 1. 安装项目

```bash
# 克隆项目
sudo git clone <your-repo-url> /opt/GenAI2OpenAI
cd /opt/GenAI2OpenAI

# 安装 Python 依赖
sudo apt install python3 python3-pip python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 配置 systemd 服务

```bash
# 复制服务文件
sudo cp deploy/genai2openai.service /etc/systemd/system/

# 编辑服务文件，替换 YOUR_TOKEN_HERE
sudo nano /etc/systemd/system/genai2openai.service

# 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable genai2openai
sudo systemctl start genai2openai

# 检查状态
sudo systemctl status genai2openai
```

## 3. 配置 Nginx

```bash
# 安装 Nginx
sudo apt install nginx

# 复制配置文件
sudo cp deploy/genai2openai.nginx.conf /etc/nginx/sites-available/genai2openai

# 编辑配置文件，替换域名和证书路径
sudo nano /etc/nginx/sites-available/genai2openai

# 创建软链接
sudo ln -s /etc/nginx/sites-available/genai2openai /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
```

## 4. SSL 证书

### 使用 Let's Encrypt (推荐)

```bash
# 安装 Certbot
sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d your-domain.com

# 自动续期
sudo crontab -e
# 添加: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 使用自签名证书 (测试)

```bash
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/your-domain.com.key \
  -out /etc/ssl/certs/your-domain.com.pem
```

## 5. 防火墙

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

## 6. 验证部署

```bash
# 检查服务状态
sudo systemctl status genai2openai

# 检查 Nginx 状态
sudo systemctl status nginx

# 测试 API
curl https://your-domain.com/health
```

## 常用命令

```bash
# 查看服务日志
sudo journalctl -u genai2openai -f

# 重启服务
sudo systemctl restart genai2openai

# 查看 Nginx 日志
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```
