"""
Create a Mailchimp *draft* campaign from a Benchmark Beaters weekly kit
(outputs/benchmark-beaters-weekly/<date>/email.html + meta.json).

Never sends: the draft is left in Mailchimp to review, fix links in and
schedule by hand.

Env vars:
  MAILCHIMP_API_KEY     required, e.g. abc123-us21 (datacenter taken from suffix)
  MAILCHIMP_LIST_ID     required, audience id
  MAILCHIMP_FROM_NAME   default "5i Research"
  MAILCHIMP_REPLY_TO    required, verified sender address
  MAILCHIMP_SEGMENT_ID  optional, saved segment / tag id for BB subscribers
"""

import argparse
import json
import os
import sys

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_ROOT = os.path.join(REPO_ROOT, "outputs", "benchmark-beaters-weekly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default="latest", help="Kit folder name under benchmark-beaters-weekly")
    args = ap.parse_args()

    key = os.environ.get("MAILCHIMP_API_KEY")
    list_id = os.environ.get("MAILCHIMP_LIST_ID")
    reply_to = os.environ.get("MAILCHIMP_REPLY_TO")
    if not (key and list_id and reply_to):
        print("Mailchimp env vars not set; skipping draft creation.")
        return
    base = f"https://{key.rsplit('-', 1)[-1]}.api.mailchimp.com/3.0"
    auth = ("anystring", key)

    kit = os.path.join(KIT_ROOT, args.kit)
    with open(os.path.join(kit, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(kit, "email.html"), encoding="utf-8") as f:
        email_html = f.read()

    recipients = {"list_id": list_id}
    if os.environ.get("MAILCHIMP_SEGMENT_ID"):
        recipients["segment_opts"] = {"saved_segment_id": int(os.environ["MAILCHIMP_SEGMENT_ID"])}

    r = requests.post(f"{base}/campaigns", auth=auth, timeout=30, json={
        "type": "regular",
        "recipients": recipients,
        "settings": {
            "title": f"Benchmark Beaters Weekly {meta['as_of']}",
            "subject_line": meta["email_subject"],
            "preview_text": meta["preview_text"],
            "from_name": os.environ.get("MAILCHIMP_FROM_NAME", "5i Research"),
            "reply_to": reply_to,
        },
    })
    r.raise_for_status()
    campaign_id = r.json()["id"]

    r = requests.put(f"{base}/campaigns/{campaign_id}/content", auth=auth, timeout=30,
                     json={"html": email_html})
    r.raise_for_status()

    print(f"Created Mailchimp draft {campaign_id}: {meta['email_subject']}")
    if "{{" in email_html:
        print("::warning::Draft still contains {{TABLE_PDF_URL}} / {{CHARTS_PDF_URL}} placeholders - "
              "replace them in Mailchimp before sending.")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print(f"Mailchimp API error: {e.response.status_code} {e.response.text}", file=sys.stderr)
        sys.exit(1)
