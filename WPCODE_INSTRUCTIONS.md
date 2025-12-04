# WordPress WPCode Integration Instructions

## How to Add the Franquicia Boost Chat Widget to WordPress

### Step 1: Install WPCode Plugin
1. Go to your WordPress admin dashboard
2. Navigate to **Plugins > Add New**
3. Search for "WPCode" (by WPCode)
4. Install and activate the plugin

### Step 2: Add the Chat Widget Code
1. Go to **Code Snippets > Add Snippet** (or **WPCode > Add Snippet**)
2. Click **"Add Your Custom Code (New Snippet)"**
3. Give it a name: "Franquicia Boost Chat Widget"
4. Select **Code Type**: **HTML**
5. Select **Location**: **Run Everywhere** (or choose specific pages)
6. Copy the entire contents of `wpcode-widget.html` file
7. Paste it into the code editor
8. Toggle **"Active"** to ON
9. Click **"Save Snippet"**

### Step 3: Verify Installation
1. Visit your WordPress site
2. You should see a floating chat button in the bottom-right corner
3. Click it to open the chat widget
4. Test by asking a question like "What franchises are available?"

## Features

- ✅ Fully self-contained (no external dependencies)
- ✅ Responsive design (works on mobile and desktop)
- ✅ Beautiful gradient design matching your brand
- ✅ AI notice banner (can be dismissed)
- ✅ Suggested prompt buttons
- ✅ Loading indicators
- ✅ Smooth animations
- ✅ WordPress-friendly (won't conflict with your theme)

## Customization

### Change API URL
If your API is hosted elsewhere, find this line in the code:
```javascript
const API_URL = 'https://fqbbot.com/api/chat';
```
Change it to your API URL.

### Change Colors
The widget uses a purple gradient. To change colors, search for:
- `#667eea` (primary purple)
- `#764ba2` (secondary purple)
- `#10b981` (green for user messages)

Replace these hex codes with your brand colors.

### Position
To change the widget position, modify:
```css
#fb-chat-widget-container {
    bottom: 20px;
    right: 20px;
}
```

## Troubleshooting

**Widget not appearing?**
- Make sure the snippet is **Active** in WPCode
- Check if your theme has a conflicting z-index
- Try clearing WordPress cache

**API errors?**
- Verify the API URL is correct
- Check browser console for errors (F12)
- Ensure CORS is enabled on your API

**Styling conflicts?**
- The widget uses `fb-` prefix for all classes to avoid conflicts
- If issues persist, wrap the code in an iframe

## Support

For issues or questions, contact support@franquiciaboost.com

