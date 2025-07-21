# Cloudflare R2 Storage Setup Guide

This guide will help you configure Cloudflare R2 storage for your GPU QuickCap application.

## Prerequisites

1. A Cloudflare account
2. Access to Cloudflare R2 (requires a paid Cloudflare plan or R2 subscription)

## Step 1: Create an R2 Bucket

1. Log in to your Cloudflare dashboard
2. Navigate to **R2 Object Storage** in the sidebar
3. Click **Create bucket**
4. Choose a unique bucket name (e.g., `quickcap-videos`)
5. Select your preferred region
6. Click **Create bucket**

## Step 2: Generate R2 API Token

1. In the Cloudflare dashboard, go to **My Profile** → **API Tokens**
2. Click **Create Token**
3. Use the **Custom token** template
4. Configure the token with these permissions:
   - **Account** - `Cloudflare R2:Edit`
   - **Zone Resources** - Include `All zones` or specific zones
   - **Account Resources** - Include `All accounts` or your specific account

5. Click **Continue to summary** and then **Create Token**
6. **Important**: Copy the token immediately as it won't be shown again

## Step 3: Get Your Account ID

1. In the Cloudflare dashboard, go to the right sidebar
2. Copy your **Account ID** from the right sidebar

## Step 4: Configure Environment Variables

Update your `.env` file in the `backend` directory with your R2 credentials:

```env
# R2 Storage Configuration
R2_ACCOUNT_ID=your_cloudflare_account_id_here
R2_ACCESS_KEY_ID=your_r2_access_key_id_here
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key_here
R2_BUCKET_NAME=your_r2_bucket_name_here
R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com
R2_PUBLIC_URL=https://your_custom_domain_or_r2_url_here
```

### Variable Explanations:

- **R2_ACCOUNT_ID**: Your Cloudflare Account ID
- **R2_ACCESS_KEY_ID**: The Access Key ID from your R2 API token
- **R2_SECRET_ACCESS_KEY**: The Secret Access Key from your R2 API token
- **R2_BUCKET_NAME**: The name of your R2 bucket (e.g., `quickcap-videos`)
- **R2_ENDPOINT_URL**: Replace `your_account_id` with your actual Account ID
- **R2_PUBLIC_URL**: (Optional) Your custom domain for R2, or leave as the default R2 URL

## Step 5: Create R2 API Credentials

1. In the Cloudflare dashboard, navigate to **R2 Object Storage**
2. Click on **Manage R2 API tokens**
3. Click **Create API token**
4. Choose permissions:
   - **Object Read & Write** for your bucket
   - **Admin Read & Write** if you want full access
5. Choose **TTL** (time to live) - select "No expiry" for production
6. Click **Create API token**
7. Copy both the **Access Key ID** and **Secret Access Key**

## Step 6: Set Up Custom Domain (Optional but Recommended)

1. In R2 dashboard, go to your bucket
2. Click **Settings** → **Custom Domains**
3. Click **Connect Domain**
4. Enter your domain (e.g., `videos.yourdomain.com`)
5. Follow the DNS configuration instructions
6. Update `R2_PUBLIC_URL` in your `.env` file with your custom domain

## Step 7: Install Dependencies

Make sure boto3 is installed:

```bash
pip install boto3==1.34.34 botocore==1.34.34
```

## Step 8: Test the Configuration

1. Start your application
2. Check the logs for R2 initialization messages
3. Use the storage status endpoint: `GET /api/storage/status`
4. Upload a video to test the integration

## API Endpoints

Once configured, you'll have access to these R2 storage endpoints:

- `GET /api/storage/status` - Check R2 storage status
- `GET /api/storage/videos` - List all videos in R2
- `GET /api/storage/videos/{video_id}` - Get specific video info
- `DELETE /api/storage/videos/{video_id}` - Delete a video from R2
- `GET /api/storage/videos/{video_id}/url` - Get video URLs (public and presigned)

## Security Best Practices

1. **Never commit your `.env` file** to version control
2. Use **least privilege** access for R2 API tokens
3. Set up **custom domains** for better performance and branding
4. Consider **CORS policies** for your R2 bucket if accessing from browser
5. Regularly **rotate your API tokens**

## Troubleshooting

### Common Issues:

1. **"Access Denied" errors**: Check your API token permissions
2. **"Bucket not found"**: Verify bucket name and region
3. **"Invalid credentials"**: Double-check your Access Key ID and Secret
4. **Connection timeouts**: Check your internet connection and R2 service status

### Debug Mode:

Enable debug logging in your application to see detailed R2 operation logs.

## Cost Considerations

- R2 charges for storage, requests, and data transfer
- First 10GB of storage per month is free
- Check current R2 pricing on Cloudflare's website
- Consider lifecycle policies for automatic cleanup of old videos

## Backup Strategy

While R2 is highly durable, consider:
- Keeping local copies of important videos
- Setting up cross-region replication if needed
- Regular backup verification

---

For more information, visit the [Cloudflare R2 documentation](https://developers.cloudflare.com/r2/).