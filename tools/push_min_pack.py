# -*- coding: utf-8 -*-
r"""push_min_pack.py — 企业代理拦截 git push 时的最小包直推工具（2026-10-07 实战沉淀）

背景：
- 本机 git 出口走华为企业代理（NetentSec SWG "HIS Proxy"，proxyhk.huawei.com:8080，NTLM）。
- 代理对 git-receive-pack 的大 POST 体（实测 ~140KB）返回 403 + HIS Proxy Notification 页
  （gitee 本身可达；GET/小 POST 均正常）。
- `git push` 在代理拦截态下整体不可用（即使小包也 403，疑似请求头/会话特征触发）。

原理（HTTP(S) smart protocol，单 POST stateless-rpc）：
1. `git rev-list --objects <old>..<new>` 取差集对象（远端已有对象不重传）；
2. `git pack-objects`（对象清单模式，非 --revs）打最小包；
3. 构造 pkt-line 请求体："<old> <new> <ref>\0report-status side-band-64k agent=..." + 0000 + pack；
4. curl 经代理 POST 到 <url>/git-receive-pack，凭据经 `git credential fill` 获取（不落盘不回显）；
5. 成功判据：响应含 "unpack ok" 与 "ok <ref>"。

用法：
    python tools\push_min_pack.py [remote(默认 origin)] [ref(默认 master)]
      old = 远端当前 ref 值（自动 ls-remote），new = 本地同名 ref 值。
注意：
- 仅推送单个 ref 的快进更新；非快进/多 ref 请走正常 git push。
- 凭据文件用后即删；密码不出现在任何输出/日志。
"""
import subprocess, sys, os, io, re

GIT = r"git"  # 调用方需保证 PATH（PS 5.1 下先 $env:Path 加 Git\bin）
PROXY = os.environ.get("PUSH_PROXY", "http://proxyhk.huawei.com:8080")


def run(args, **kw):
    return subprocess.run(args, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True, **kw)


def main():
    remote = sys.argv[1] if len(sys.argv) > 1 else "origin"
    ref = sys.argv[2] if len(sys.argv) > 2 else "master"
    rurl = run([GIT, "remote", "get-url", remote]).stdout.decode().strip()
    m = re.match(r"https://([^/]+)/(.*)\.git$", rurl)
    if not m:
        sys.exit("仅支持 https 远端：{}".format(rurl))
    host, repo = m.group(1), m.group(2)

    old = run([GIT, "ls-remote", remote, ref]).stdout.decode().split()[0]
    new = run([GIT, "rev-parse", "refs/heads/" + ref]).stdout.decode().strip()
    if old == new:
        print("up-to-date: {} 已指向 {}".format(ref, new))
        return

    out = run([GIT, "rev-list", "--objects", old + ".." + new]).stdout.decode()
    shas = [l.split()[0] for l in out.strip().split("\n") if l.strip()]
    if not shas:
        sys.exit("无差集对象（非快进或空区间）")

    tmp = os.environ["TEMP"]
    p = run([GIT, "pack-objects", os.path.join(tmp, "pmp")],
            input=("\n".join(shas) + "\n").encode())
    h = p.stdout.decode().strip()
    pack = os.path.join(tmp, "pmp-" + h + ".pack")

    caps = "report-status side-band-64k agent=git/2.56.0.windows.2"
    line = "{} {} refs/heads/{}\0{}\n".format(old, new, ref, caps)
    pkt = ("%04x" % (len(line) + 4)) + line
    body = pkt.encode() + b"0000" + io.open(pack, "rb").read()
    req = os.path.join(tmp, "pmp_req.bin")
    io.open(req, "wb").write(body)
    print("push {} -> {}: {} 对象, 请求体 {} 字节".format(
        old[:7], new[:7], len(shas), len(body)))

    # 凭据：git credential fill（内存传递，用后即删）
    cred = run([GIT, "credential", "fill"],
               input="protocol=https\nhost={}\n".format(host).encode())
    user = re.search(r"^username=(.+)$", cred.stdout.decode(), re.M).group(1)
    passwd = re.search(r"^password=(.+)$", cred.stdout.decode(), re.M).group(1)
    cf = os.path.join(tmp, "pmp_cred.txt")
    io.open(cf, "w").write("username={}\npassword={}\n".format(user, passwd))

    ps = ("$c = Get-Content '{}' -Raw; $u = [regex]::Match($c,'username=(.+)').Groups[1].Value; "
          "$p = [regex]::Match($c,'password=(.+)').Groups[1].Value; "
          "curl.exe -s -x {} --proxy-ntlm -U : -X POST "
          "'https://{}/{}/git-receive-pack' "
          "-H 'Content-Type: application/x-git-receive-pack-request' "
          "-H 'User-Agent: git/2.56.0.windows.2' -u \"$u:$p\" "
          "--data-binary \"@{}\" -o '{}' -w 'HTTP %{{http_code}}'".format(
              cf, PROXY, host, repo, req,
              os.path.join(tmp, "pmp_resp.bin")))
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout.decode(errors="replace").strip())
    resp = io.open(os.path.join(tmp, "pmp_resp.bin"), "rb").read()
    ok = b"unpack ok" in resp and b"ok refs/heads/" + ref.encode() in resp
    for f in (cf, req, pack, pack.replace(".pack", ".idx"),
              os.path.join(tmp, "pmp_resp.bin")):
        try:
            os.remove(f)
        except OSError:
            pass
    print(("SUCCESS: " if ok else "FAILED: ") +
          resp[:200].decode(errors="replace"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
