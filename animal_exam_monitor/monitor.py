from __future__ import annotations

import json
import os
import re
import smtplib
import ssl
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://kjt.hubei.gov.cn"
LIST_URL = f"{BASE_URL}/sydw/portal/exam/toListUi"
HOME_URL = f"{BASE_URL}/sydw/"
STATE_PATH = Path(__file__).with_name("state.json")

# General experimental-animal qualification notices usually contain one or more of
# these phrases. We still record every new item on the official ability-evaluation
# page, but use this list to mark the most likely '抢名额' notices as HIGH priority.
HIGH_PRIORITY_TERMS = (
    "实验动物从业人员",
    "动物实验从业人员",
    "从业人员技术咨询",
    "能力辅导评价",
    "能力提升及评价",
    "能力评价",
)

LOWER_PRIORITY_TERMS = (
    "设施运行维护",
    "设施管理人员",
    "饲养管理人员",
    "饲养人员",
    "斑马鱼",
)

DETAIL_RE = re.compile(r"/sydw/portal/exam/details\?id=([0-9a-fA-F-]{20,})")


@dataclass(frozen=True)
class Notice:
    id: str
    title: str
    url: str
    priority: str


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def priority_for(title: str) -> str:
    if any(term in title for term in HIGH_PRIORITY_TERMS) and not any(
        term in title for term in LOWER_PRIORITY_TERMS
    ):
        return "HIGH"
    if "从业人员" in title or "能力" in title:
        return "MEDIUM"
    return "LOW"


def extract_notices(html: str, base_url: str = LIST_URL) -> list[Notice]:
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, Notice] = {}

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        match = DETAIL_RE.search(href)
        if not match:
            continue
        notice_id = match.group(1)
        title = normalize_text(a.get_text(" ", strip=True))
        if not title:
            title = normalize_text(a.parent.get_text(" ", strip=True) if a.parent else "")
        if not title:
            title = f"湖北省实验动物能力提升通知 {notice_id}"
        url = urljoin(base_url, href)
        found[notice_id] = Notice(notice_id, title, url, priority_for(title))

    if not found:
        for match in DETAIL_RE.finditer(html):
            notice_id = match.group(1)
            start = max(0, match.start() - 300)
            end = min(len(html), match.end() + 500)
            nearby = BeautifulSoup(html[start:end], "html.parser").get_text(" ", strip=True)
            nearby = normalize_text(nearby)
            title = nearby[:180] or f"湖北省实验动物能力提升通知 {notice_id}"
            url = f"{BASE_URL}/sydw/portal/exam/details?id={notice_id}"
            found[notice_id] = Notice(notice_id, title, url, priority_for(title))

    return list(found.values())


def fetch_static(session: requests.Session, url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
        )
    }
    resp = session.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding
    return resp.text


def fetch_rendered(url: str) -> str:
    """Render the JS-driven list page with headless Chrome."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,1600")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
    )

    driver = webdriver.Chrome(options=options)
    try:
        driver.set_page_load_timeout(30)
        driver.get(url)

        def page_has_data(drv) -> bool:
            source = drv.page_source
            return bool(DETAIL_RE.search(source)) or "能力提升及评价" in source

        try:
            WebDriverWait(driver, 15).until(page_has_data)
        except Exception:
            time.sleep(3)
        return driver.page_source
    finally:
        driver.quit()


def collect_notices() -> list[Notice]:
    session = requests.Session()
    notices: dict[str, Notice] = {}
    errors: list[str] = []

    for url in (LIST_URL, HOME_URL):
        try:
            html = fetch_static(session, url)
            for n in extract_notices(html, url):
                notices[n.id] = n
        except Exception as exc:
            errors.append(f"static {url}: {exc!r}")

    if not notices:
        try:
            html = fetch_rendered(LIST_URL)
            for n in extract_notices(html, LIST_URL):
                notices[n.id] = n
        except Exception as exc:
            errors.append(f"rendered {LIST_URL}: {exc!r}")

    if not notices:
        raise RuntimeError(
            "No exam notices could be extracted from the official site. "
            + " | ".join(errors)
        )

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return sorted(notices.values(), key=lambda n: (order[n.priority], n.title))


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"initialized": False, "seen_ids": [], "last_event_utc": None}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"initialized": False, "seen_ids": [], "last_event_utc": None}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def github_issue(title: str, body: str) -> None:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if not token or not repo:
        print("[warn] GITHUB_TOKEN/GITHUB_REPOSITORY missing; skipping GitHub issue")
        return

    api = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.post(api, headers=headers, json={"title": title, "body": body}, timeout=20)
    resp.raise_for_status()
    print(f"[notify] GitHub issue created: {resp.json().get('html_url')}")


def pushplus(title: str, body: str) -> None:
    token = os.getenv("PUSHPLUS_TOKEN", "").strip()
    if not token:
        return
    resp = requests.post(
        "https://www.pushplus.plus/send",
        json={"token": token, "title": title, "content": body, "template": "html"},
        timeout=20,
    )
    resp.raise_for_status()
    print("[notify] PushPlus message sent")


def email_alert(title: str, body: str) -> None:
    """Send a direct SMTP email when repository secrets are configured.

    Defaults target QQ Mail, but all SMTP values are configurable through secrets.
    No address, password or auth code is stored in the public repository.
    """
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_AUTH_CODE", "").strip()
    recipient = os.getenv("ALERT_EMAIL", "").strip()
    if not smtp_user or not smtp_password or not recipient:
        return

    host = os.getenv("SMTP_HOST", "smtp.qq.com").strip() or "smtp.qq.com"
    port = int(os.getenv("SMTP_PORT", "465").strip() or "465")

    msg = EmailMessage()
    msg["Subject"] = title
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as smtp:
        smtp.login(smtp_user, smtp_password)
        smtp.send_message(msg)
    print("[notify] Direct email sent")


def run_channel(name: str, func: Callable[[str, str], None], title: str, body: str) -> None:
    """Do not let one failed notification channel suppress the others or cause spam."""
    try:
        func(title, body)
    except Exception as exc:
        print(f"[warn] notification channel {name} failed: {exc!r}", file=sys.stderr)


def notify_new(notices: Iterable[Notice]) -> None:
    for notice in notices:
        icon = "🚨" if notice.priority == "HIGH" else "🔔"
        title = f"{icon} 湖北实验动物考试新通知：{notice.title[:80]}"
        body = (
            "发现新的湖北省实验动物能力提升/评价通知。\n\n"
            f"优先级：{notice.priority}\n"
            f"标题：{notice.title}\n"
            f"官方链接：{notice.url}\n"
            f"发现时间（UTC）：{datetime.now(timezone.utc).isoformat()}\n\n"
            "建议立即打开湖北省实验动物公共服务平台并尝试报名，不要等第二次提醒。\n\n"
            "@wenzekan80-gif"
        )
        run_channel("github", github_issue, title, body)
        run_channel("email", email_alert, title, body)
        run_channel("pushplus", pushplus, title, body.replace("\n", "<br>"))


def send_test_notification() -> None:
    title = "✅ 湖北实验动物考试监控：测试提醒"
    body = (
        "监控工作流已成功运行。这是一条测试通知。\n\n"
        f"官方能力提升页面：{LIST_URL}\n\n"
        "以后发现新的通知时，会自动创建 GitHub Issue；如果配置了 SMTP secrets，"
        "会同步直接发送邮件；如果配置 PUSHPLUS_TOKEN，也会同步推送到微信。\n\n"
        "@wenzekan80-gif"
    )
    run_channel("github", github_issue, title, body)
    run_channel("email", email_alert, title, body)
    run_channel("pushplus", pushplus, title, body.replace("\n", "<br>"))


def main() -> int:
    if os.getenv("TEST_NOTIFICATION", "").lower() in {"1", "true", "yes"}:
        send_test_notification()
        return 0

    state = load_state()
    current = collect_notices()
    current_ids = {n.id for n in current}
    seen = set(state.get("seen_ids", []))

    print(f"[info] extracted {len(current)} official notices")
    for n in current[:20]:
        print(f"[{n.priority}] {n.id} | {n.title} | {n.url}")

    now = datetime.now(timezone.utc).isoformat()

    # First run is a baseline only. This prevents a flood of old notices immediately
    # after the workflow is installed.
    if not state.get("initialized"):
        state = {
            "initialized": True,
            "seen_ids": sorted(current_ids),
            "last_event_utc": now,
        }
        save_state(state)
        print("[info] baseline created; no alert sent on first run")
        return 0

    new_items = [n for n in current if n.id not in seen]
    if not new_items:
        # Do not rewrite state.json on every poll. That would create hundreds of
        # meaningless commits per day when the workflow persists state.
        print("[info] no new notices")
        return 0

    print(f"[alert] {len(new_items)} new notice(s) found")
    notify_new(new_items)

    state["seen_ids"] = sorted(seen | current_ids)
    state["last_event_utc"] = now
    save_state(state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[fatal] {exc!r}", file=sys.stderr)
        raise
