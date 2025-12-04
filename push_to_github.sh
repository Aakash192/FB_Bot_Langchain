#!/bin/bash
cd /home/ubuntu/FB-Bot_Python

echo "Adding files..."
git add -A

echo "Current status:"
git status --short

echo ""
echo "Committing changes..."
git commit -m "Fix Q&A matcher - stricter algorithm to prevent false matches

- Increased similarity threshold to 80%
- Added franchise name detection 
- Filter stopwords
- Use Jaccard similarity
- Prevent FDD questions from matching generic Q&A
- Added documentation and restart scripts"

echo ""
echo "Pushing to GitHub..."
git push origin main

echo ""
echo "Done! Check https://github.com/Aakash192/FB_Bot_Langchain.git"

