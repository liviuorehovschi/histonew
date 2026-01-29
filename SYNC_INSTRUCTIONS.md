# GitHub ↔ Hugging Face Space sync

## Current setup

- **origin** = GitHub: `https://github.com/liviuorehovschi/histonew.git`
- **hf** = Hugging Face Space: `https://huggingface.co/spaces/liviuorehovschi/histonew`
- **old-origin** = previous HF Space (histomancer), kept for reference

Pushes to **GitHub main** trigger a GitHub Action that pushes **main → HF Space main** (Space redeploys automatically).

---

## What you need to do manually

### 1. Push local main to GitHub (if not already done)

From your local repo (Desktop/histonew), run:

```bash
git checkout main
git push origin main --force
```

This makes GitHub `main` match your HF production (and local main).  
Optional: push the backup branch so it exists on GitHub:

```bash
git push -u origin backup/local-before-hf-import
```

### 2. Add GitHub Secrets for the deploy workflow

In GitHub: **Repo → Settings → Secrets and variables → Actions**.

Add two repository secrets:

| Name        | Value |
|------------|--------|
| `HF_TOKEN` | Your Hugging Face **write** token (Settings → Access Tokens on [huggingface.co](https://huggingface.co); create a token with **Write** access). |
| `HF_USERNAME` | Your Hugging Face username (e.g. `liviuorehovschi`). |

Without these, the "Deploy to Hugging Face Space" workflow will fail when it tries to push to the Space.

### 3. Commit and push the workflow file

If the workflow file is not yet on GitHub:

```bash
git add .github/workflows/deploy_to_hf_space.yml
git commit -m "Add GitHub Actions: deploy main to HF Space on push"
git push origin main
```

---

## Verify

1. After a push to `main`, open **Actions** on GitHub and confirm the "Deploy to Hugging Face Space" run succeeds.
2. On the [Space page](https://huggingface.co/spaces/liviuorehovschi/histonew), check that **Last update** changed and the Space is rebuilding.
3. If the workflow fails, open the failed run, check the logs of the "Add HF remote and push" step (e.g. auth or permission errors), then fix the secret or token permissions.

---

## Backup branches (no action needed)

- **backup/local-before-hf-import** – local state before this sync setup (push to origin if you want it on GitHub).
- **backup/github-main-before-sync** – previous GitHub `main` before it was overwritten by HF production; already pushed to `origin`.
