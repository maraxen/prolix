---
title: "Common Commands"
description: "Engaging cluster login, workspace sync, submit, and queue commands moved verbatim from CLAUDE.md."
status: reference
date: '261007'
---

**Common Commands:**
```bash
just -g cluster-login engaging                          # SSH control master
just -g cluster-push-workspace prolix engaging          # Sync workspace
just -g cluster-submit prolix script.sh                 # Submit job
just -g cluster-queue engaging                          # View queue
```
