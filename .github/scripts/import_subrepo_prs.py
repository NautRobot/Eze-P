import os
import sys
import subprocess
from github import Github, GithubIntegration
from git import Repo

def run(cmd, **kwargs):
    print(f">> {cmd}")
    subprocess.check_call(cmd, shell=True, **kwargs)

def main():
    token       = os.environ["GITHUB_TOKEN"]
    prefix      = os.environ["SUBPREFIX"]
    subrepo     = os.environ["SUBREPO"]
    upstream    = os.environ["UPSTREAM"]
    target      = os.environ["TARGET"]
    pr_list     = os.environ["PR_LIST"].split(",")
    repo_path   = os.getcwd()
    repo        = Repo(repo_path)

    gh = Github(token)
    super_repo = gh.get_repo(os.environ["GITHUB_REPOSITORY"])
    sub_repo   = gh.get_repo(upstream)

    # Ensure we have target branch locally
    run(f"git fetch origin {target}")
    run(f"git checkout {target}")

    for pr_num in pr_list:
        pr_num = pr_num.strip()
        pr = sub_repo.get_pull(int(pr_num))

        title   = pr.title
        body    = pr.body or ""
        head_ref= pr.head.ref
        head_url= pr.head.repo.clone_url
        is_draft= pr.draft
        author  = pr.user.login

        # Create a sanitized branch name
        tclean = target.replace("/", "_")
        src_clean = subrepo.replace("/", "_")
        branch = f"import/{tclean}/{src_clean}/pr-{pr_num}"

        # Checkout new branch
        run(f"git checkout -b {branch}")

        # Pull in the subtree from the PR head
        run(f"git subtree pull --prefix={prefix} {head_url} {head_ref}")

        # Push the branch
        run(f"git push origin {branch}")

        # Build PR body
        footer = (
            f"\n\n---\n"
            f"🔁 Imported from [{upstream}#{pr_num}](https://github.com/{upstream}/pull/{pr_num})\n"
            f"🧑‍💻 Originally authored by @{author}"
        )
        full_body = body + footer

        # Create the PR on the super‐repo
        super_repo.create_pull(
            title=title,
            body=full_body,
            head=branch,
            base=target,
            draft=is_draft,
        )

        # label it
        pr_created = super_repo.get_pulls(head=f"{repo.remote().url.split('/')[-1].replace('.git','')}:{branch}", 
                                          state="open")[0]
        pr_created.add_to_labels("imported pr")

        # checkout back to target before next iteration
        run(f"git checkout {target}")

if __name__ == "__main__":
    main()
