# GitHub Upload Command Record

Date: 2026-06-23

Source directory:

```bash
SRC=/projects/u6ls/ivy77/lhm-hand-new
STAGE=/projects/u6ls/ivy77/oasis_github_private_stage
```

## Commands actually executed in this environment

GitHub CLI preflight was attempted but the CLI is not installed in the current environment:

```bash
gh auth status
# result: /usr/bin/bash: gh: command not found

gh repo list --limit 20
# result: /usr/bin/bash: gh: command not found
```

Credential checks performed without printing secrets:

```bash
command -v gh || command -v hub || command -v git
printf 'GITHUB_TOKEN=%s\nGH_TOKEN=%s\n' "${GITHUB_TOKEN:+set}" "${GH_TOKEN:+set}"
test -f ~/.config/gh/hosts.yml && echo gh_hosts_present || echo gh_hosts_absent
test -f ~/.git-credentials && echo git_credentials_present || echo git_credentials_absent
test -f ~/.netrc && echo netrc_present || echo netrc_absent
git credential fill <<'CRED'
protocol=https
host=github.com

CRED
awk 'tolower($1)=="machine" {print $2}' ~/.netrc 2>/dev/null | sed 's/.*github.*/github.com-entry/' | sort -u
```

Observed status:

- `gh` is not available.
- `GITHUB_TOKEN` and `GH_TOKEN` are not set.
- No `~/.config/gh/hosts.yml` or `~/.git-credentials` was found.
- `~/.netrc` exists but only reports `api.wandb.ai`, not GitHub.

Staging directory creation:

```bash
rm -rf /projects/u6ls/ivy77/oasis_github_private_stage
mkdir -p /projects/u6ls/ivy77/oasis_github_private_stage
```

Candidate scan:

```bash
cd /projects/u6ls/ivy77/lhm-hand-new
find . -type f -name '*.py' \
  ! -path './logs/*' \
  ! -path './log/*' \
  ! -path './output/*' \
  ! -path './outputs/*' \
  ! -path './results/*' \
  ! -path './exp/*' \
  ! -path './experiments/*' \
  ! -path './runs/*' \
  ! -path './wandb/*' \
  ! -path './assets/*' \
  ! -path './data/*' \
  ! -path './datasets/*' \
  ! -path './checkpoints/*' \
  ! -path './checkpoint/*' \
  ! -path './pretrained/*' \
  ! -path './pretrained_models/*' \
  ! -path './example_data/*' \
  ! -path '*/__pycache__/*' \
  | sort > /tmp/oasis_py_candidates.txt
```

Manual selected-file list generation:

```bash
cd /projects/u6ls/ivy77/lhm-hand-new
{
  printf '%s\n' \
    ./app.py \
    ./app_motion.py \
    ./editing_on_hand.py \
    ./render_piano.py \
    ./train_in_co.py \
    ./train_interhand.py \
    ./train_mix_3.py \
    ./finetune_edit.py \
    ./finetune_edit_ohta.py \
    ./finetune_edit_wild_ohta.py \
    ./finetune_interhand.py \
    ./finetune_interhand_ohta.py \
    ./finetune_wild.py \
    ./finetune_wild_ohta.py \
    ./finetune_wild_id2_ohta.py \
    ./generate_handavatar_test_anno.py \
    ./generate_interhand_anno.py \
    ./test_hand.py \
    ./test_handavatar.py \
    ./test_hanco.py \
    ./test_wild.py
  find ./LHM/datasets ./LHM/losses ./LHM/outputs ./LHM/utils -type f -name '*.py'
  find ./LHM/runners -type f -name '*.py' ! -name '*copy*' ! -name '*alice*' ! -name '*ori.py'
  find ./LHM/models -maxdepth 1 -type f -name '*.py' ! -name '*_ori.py' ! -name '*ori.py'
  find ./LHM/models/encoders -maxdepth 1 -type f -name '*.py'
  find ./LHM/models/rendering -maxdepth 1 -type f -name '*.py' ! -name '*_ori.py' ! -name '*ori.py'
  find ./LHM/models/rendering/smplx_gsavatar ./LHM/models/rendering/utils ./LHM/models/styleunet -type f -name '*.py'
  find ./scene -type f -name '*.py'
  find ./splatformer/utils -type f -name '*.py'
  find ./engine -maxdepth 1 -type f -name '*.py'
  find ./engine/SegmentAPI ./engine/pose_estimation/blocks ./engine/pose_estimation/pose_utils -type f -name '*.py'
  find ./engine/pose_estimation -maxdepth 1 -type f -name '*.py'
  find ./tools/libcore ./tools/metrics ./tools/model/smplx -type f -name '*.py'
  find ./tools_utils -maxdepth 1 -type f -name '*.py'
  find ./tools_utils/libcore ./tools_utils/metrics ./tools_utils/model/ohta/configs ./tools_utils/model/smplx ./tools_utils/model/smplx_ours -type f -name '*.py'
  find ./tools_utils/model/handavatar/configs ./tools_utils/model/handavatar/core -type f -name '*.py'
  printf '%s\n' \
    ./tools_utils/model/handavatar/__init__.py \
    ./tools_utils/model/handavatar/run_interhand.py \
    ./tools_utils/model/handavatar/train.py
} | sort -u > /tmp/oasis_selected_py_files.txt
```

Copy selected files while preserving relative paths:

```bash
SRC=/projects/u6ls/ivy77/lhm-hand-new
STAGE=/projects/u6ls/ivy77/oasis_github_private_stage
cd "$SRC"
while IFS= read -r f; do
  mkdir -p "$STAGE/$(dirname "$f")"
  cp "$SRC/$f" "$STAGE/$f"
done < /tmp/oasis_selected_py_files.txt
sed 's#^\./##' /tmp/oasis_selected_py_files.txt > "$STAGE/UPLOAD_FILE_LIST.txt"
```

Safety checks executed:

```bash
cd "$STAGE"
find . -type f | sort
find . -type f ! -name '*.py' ! -name 'README.md' ! -name '.gitignore' ! -name 'UPLOAD_FILE_LIST.txt' ! -name 'GITHUB_UPLOAD_COMMANDS.md' ! -name 'GITHUB_RELEASE_PLAN.md' | sort
find . -type f -size +2M -print
grep -RInE "(token|password|secret|api_key|Authorization|/projects/u6ls/ivy77|/home/|/lus/|/gpfs/)" . --include='*.py' || true
```

Notes:

- The broad `token` grep matches normal model variables such as attention tokens; these are not credentials.
- Additional focused grep for real sensitive patterns and local absolute paths passed after staging-only sanitization.

Staging-only sanitization executed:

```bash
STAGE=/projects/u6ls/ivy77/oasis_github_private_stage
find "$STAGE" -type f -name '*.py' -exec perl -pi -e 's/# wandb\.login\(key\s*=\s*'"'"'[^'"'"']+'"'"'\)/# wandb.login(key=os.getenv("WANDB_API_KEY"))  # configure locally/g; s#\Q/home/z/zh174/LHM/data/\E#./wandb/#g; s#\Q/home/z/zh174/interhand\E#./data/interhand#g; s#\Q/home/z/zh174/hanco_\E#./data/hanco#g; s#\Q/home/z/zh174/hanco\E#./data/hanco#g; s#\Q/home/z/zh174/Hands11k/processed_test\E#./data/hands11k/processed_test#g; s#\Q/home/z/zh174/ohta/example_data/interhand2.6m\E#./example_data/interhand2.6m#g; s#\Q/projects/u6ls/ivy77/interhand\E#./data/interhand#g; s#\Q/home/z/zh174/miniconda3\E#LOCAL_CONDA#g; s#/gpfs/home/z/zh174#LOCAL_GPFS_HOME#g; s#/home/z/zh174#LOCAL_HOME#g; s#/projects/u6ls/ivy77#LOCAL_PROJECT_ROOT#g; s#/lus/[A-Za-z0-9_./-]+#LOCAL_LUS_PATH#g' {} +
find "$STAGE" -type f -name '*.py' -exec sed -i -E '/^[[:space:]]*#[[:space:]]*api_key[[:space:]]*=/d; /^[[:space:]]*#[[:space:]]*wandb\.login\(key=api_key\)/d' {} +
perl -pi -e 's#/home/\.\.\./hanco_#./data/hanco#g' "$STAGE/test_hanco.py"
rg -n "c2827ee38bbb0698e36a442d38cb1b632c2463bd|/home/|/gpfs/|/projects/u6ls/ivy77|/lus/|api_key|Authorization|password|secret" "$STAGE" -g '*.py' || true
```

## Commands to run after GitHub CLI is installed and authenticated

Prefer the MOVA-hand organization if the authenticated account has permission. If not, use the authenticated user returned by `gh api user -q .login`.

```bash
gh auth status
gh repo list --limit 20
OWNER=MOVA-hand
REPO_NAME=oasis-hand-code

# If oasis-hand-code already exists, switch to:
# REPO_NAME=oasis-hand-code-private

gh repo create "$OWNER/$REPO_NAME" \
  --private \
  --description "OASIS hand avatar reconstruction code private staging repository" \
  --homepage "https://mova-hand.github.io/MOVA/"

gh repo view "$OWNER/$REPO_NAME" --json visibility -q .visibility
```

If visibility is not `PRIVATE`, stop before pushing and run:

```bash
gh repo edit "$OWNER/$REPO_NAME" --visibility private
gh repo view "$OWNER/$REPO_NAME" --json visibility -q .visibility
```

Initialize, commit, and push staging:

```bash
cd /projects/u6ls/ivy77/oasis_github_private_stage
git init
git branch -M main
git add .
git status --short
git commit -m "Initial private staging release with selected Python files"
git remote add origin "https://github.com/$OWNER/$REPO_NAME.git"
gh repo view "$OWNER/$REPO_NAME" --json visibility -q .visibility
git push -u origin main
```

Confirm the repository remains private:

```bash
gh repo view "$OWNER/$REPO_NAME" --json nameWithOwner,visibility,url
```

## MOVA project page Code link update command

Only run this after the private repository exists and visibility is confirmed as `PRIVATE`.

```bash
cd /projects/u6ls/ivy77/Academic-project-page-template
# Edit only the Code button href in index.html to:
# https://github.com/$OWNER/$REPO_NAME
git diff -- index.html
git add index.html
git commit -m "Update Code link to private GitHub repository"
git push origin master
```

## Standard flow for adding more files later

```bash
SRC=/projects/u6ls/ivy77/lhm-hand-new
STAGE=/projects/u6ls/ivy77/oasis_github_private_stage
cd "$SRC"
# Update /tmp/oasis_selected_py_files.txt with newly approved Python files only.
while IFS= read -r f; do
  mkdir -p "$STAGE/$(dirname "$f")"
  cp "$SRC/$f" "$STAGE/$f"
done < /tmp/oasis_selected_py_files.txt
sed 's#^\./##' /tmp/oasis_selected_py_files.txt > "$STAGE/UPLOAD_FILE_LIST.txt"
cd "$STAGE"
find . -type f -size +2M -print
rg -n "c2827ee38bbb0698e36a442d38cb1b632c2463bd|/home/|/gpfs/|/projects/u6ls/ivy77|/lus/|api_key|Authorization|password|secret" . -g '*.py' || true
gh repo view "$OWNER/$REPO_NAME" --json visibility -q .visibility
git add .
git status --short
git commit -m "Add selected cleaned Python files"
git push
```

## Final alignment after `.gitignore`

The `.gitignore` intentionally blocks directories named `data/`, `datasets/`, and `outputs/`. After the first local commit, the following ignored Python-only code directories were removed from the staging working tree so the directory contents, upload list, and future pushed commit are aligned with the exclusion policy:

```bash
cd /projects/u6ls/ivy77/oasis_github_private_stage
rm -rf LHM/datasets LHM/outputs tools_utils/model/handavatar/core/data
git ls-files '*.py' | sort > UPLOAD_FILE_LIST.txt
git add UPLOAD_FILE_LIST.txt GITHUB_UPLOAD_COMMANDS.md
git commit --amend --no-edit
```

Final local commit contains 292 Python files plus `.gitignore`, `README.md`, `UPLOAD_FILE_LIST.txt`, `GITHUB_UPLOAD_COMMANDS.md`, and `GITHUB_RELEASE_PLAN.md`.

## Actual GitHub repository created in this run

Because the authenticated account `ivyyy77` does not have write/create permission in `MOVA-hand` (`MOVA-hand/MOVA` reports `viewerPermission=READ`, and org membership check returned 404), the private staging repository was created under the current authenticated user:

```bash
gh auth status
gh api user -q .login
gh repo view MOVA-hand/MOVA --json nameWithOwner,visibility,isPrivate,viewerPermission,url
gh api user/memberships/orgs/MOVA-hand
gh repo view ivyyy77/oasis-hand-code --json nameWithOwner,visibility,isPrivate,url
gh repo create ivyyy77/oasis-hand-code --private \
  --description "OASIS hand avatar reconstruction code private staging repository" \
  --homepage "https://mova-hand.github.io/MOVA/"
gh repo view ivyyy77/oasis-hand-code --json nameWithOwner,visibility,isPrivate,url
```

Verified repository:

- URL: https://github.com/ivyyy77/oasis-hand-code
- Visibility: PRIVATE
- Owner: ivyyy77

Push commands used:

```bash
cd /projects/u6ls/ivy77/oasis_github_private_stage
gh auth setup-git
git remote add origin https://github.com/ivyyy77/oasis-hand-code.git
gh repo view ivyyy77/oasis-hand-code --json visibility -q .visibility
git push -u origin main
```

## MOVA project page link update attempt

The local project page repository is `MOVA-hand/MOVA` on branch `master`. The Code button was updated locally to point to the private staging repository:

```bash
cd /projects/u6ls/ivy77/Academic-project-page-template
# staged only the one-line Code href change in index.html
git commit -m "Update Code link to private GitHub repository"
git push origin master
```

Local page commit created:

```text
d940639 Update Code link to private GitHub repository
```

Push result:

```text
remote: Permission to MOVA-hand/MOVA.git denied to ivyyy77.
fatal: unable to access 'https://github.com/MOVA-hand/MOVA.git/': The requested URL returned error: 403
```

Reason: `gh repo view MOVA-hand/MOVA --json viewerPermission` reported `READ` for the current account `ivyyy77`, so GitHub rejected pushing the page commit. Grant write/admin access to `ivyyy77` on `MOVA-hand/MOVA`, then run:

```bash
cd /projects/u6ls/ivy77/Academic-project-page-template
git push origin master
```
