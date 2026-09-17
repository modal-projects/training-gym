"""Deploy a frontend preview for a PR.

Uploads the given artifact (a .tar.gz of the built frontend) to the preview
volume and deploys it for the given PR.

Run with plain `python`, not `modal run`:

    python scripts/previews/deploy.py 244 dashboard path/to/dashboard.tar.gz
"""

import argparse
import secrets
from pathlib import Path
import modal
import asyncio


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pr_number", type=int, help="PR number to deploy the preview for"
    )
    parser.add_argument(
        "type", choices=["dashboard", "docs"], help="which frontend to deploy"
    )
    parser.add_argument("artifact", type=Path, help="path to the artifact .tar.gz")
    parser.add_argument(
        "--api-url",
        default=None,
        help="dashboard backend this preview's /api should proxy to; "
        "defaults to the deployed dashboard",
    )
    args = parser.parse_args()

    if not args.artifact.is_file():
        parser.error(f"artifact not found: {args.artifact}")

    artifact_name = f"pr{args.pr_number}-{args.type}-{secrets.token_hex(8)}.tar.gz"

    deploy_preview = modal.Function.from_name("training-gym-previews", "deploy_preview")
    vol = modal.Volume.from_name("preview-artifacts")

    print(f"Uploading {args.artifact} as {artifact_name}")
    async with vol.batch_upload.aio() as batch:
        batch.put_file(str(args.artifact), f"/{artifact_name}")

    print(f"Deploying {args.type} preview for #{args.pr_number}")
    # The deployed previews app is main's code (see previews-app.yml), so on a
    # branch that introduces `api_url` it doesn't take one yet: fall back to a
    # preview proxied at the deployed dashboard rather than failing the job.
    kwargs = {"api_url": args.api_url} if args.api_url else {}
    try:
        await deploy_preview.remote.aio(
            args.pr_number, args.type, artifact_name, **kwargs
        )
    except TypeError as exc:
        if not kwargs or "api_url" not in str(exc):
            raise
        print(f"Previews app does not accept api_url yet ({exc}); deploying without it")
        await deploy_preview.remote.aio(args.pr_number, args.type, artifact_name)


if __name__ == "__main__":
    asyncio.run(main())
