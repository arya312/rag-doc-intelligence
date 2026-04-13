from dotenv import load_dotenv
load_dotenv()

import json
import http.server
import socketserver
import os

def generate_dashboard(eval_file: str = "eval_results.json") -> str:
    with open(eval_file) as f:
        data = json.load(f)

    rows = ""
    for r in data["results"]:
        h = r["hallucination"]
        ret = r["retrieval"]
        verdict_color = {"GROUNDED": "#2a7a2a", "PARTIAL": "#b87a00", "HALLUCINATED": "#cc0000"}[h["verdict"]]
        quality_color = {"GOOD": "#2a7a2a", "FAIR": "#b87a00", "POOR": "#cc0000"}[ret["quality"]]
        pages = ", ".join(str(p) for p in r["pages"])
        answer_short = r["answer"][:180].replace("<", "&lt;").replace(">", "&gt;") + "..."

        rows += f"""
        <tr>
            <td style="padding:12px;border-bottom:1px solid #eee;max-width:200px">{r["question"]}</td>
            <td style="padding:12px;border-bottom:1px solid #eee;font-size:12px;color:#666">{answer_short}</td>
            <td style="padding:12px;border-bottom:1px solid #eee;text-align:center">p.{pages}</td>
            <td style="padding:12px;border-bottom:1px solid #eee;text-align:center">
                <span style="color:{verdict_color};font-weight:600">{h["verdict"]}</span>
                <br><small style="color:#999">{h["max_similarity"]}</small>
            </td>
            <td style="padding:12px;border-bottom:1px solid #eee;text-align:center">
                <span style="color:{quality_color};font-weight:600">{ret["quality"]}</span>
                <br><small style="color:#999">{ret["avg_relevance"]}</small>
            </td>
        </tr>"""

    grounded_pct = round(data["grounded"] / data["total_questions"] * 100)
    retrieval_pct = round(data["avg_retrieval_score"] * 100)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RAG Evaluation Dashboard</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 0 auto; padding: 2rem; background: #f9f9f9; }}
        h1 {{ font-size: 24px; font-weight: 600; margin-bottom: 4px; }}
        .subtitle {{ color: #666; margin-bottom: 2rem; }}
        .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 2rem; }}
        .card {{ background: white; border-radius: 12px; padding: 1.25rem; border: 1px solid #eee; }}
        .card-label {{ font-size: 12px; color: #999; margin-bottom: 4px; }}
        .card-value {{ font-size: 28px; font-weight: 600; }}
        .card-sub {{ font-size: 12px; color: #666; margin-top: 4px; }}
        table {{ width: 100%; background: white; border-radius: 12px; border-collapse: collapse; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        th {{ padding: 12px; background: #f4f4f4; text-align: left; font-size: 13px; color: #666; font-weight: 500; }}
        tr:hover {{ background: #fafafa; }}
        .bar {{ height: 8px; border-radius: 4px; background: #eee; margin-top: 6px; }}
        .bar-fill {{ height: 100%; border-radius: 4px; background: #0066cc; }}
    </style>
</head>
<body>
    <h1>RAG Evaluation Dashboard</h1>
    <p class="subtitle">Collection: <strong>{data["collection"]}</strong> &nbsp;|&nbsp; {data["total_questions"]} questions evaluated</p>

    <div class="cards">
        <div class="card">
            <div class="card-label">Grounded answers</div>
            <div class="card-value" style="color:#2a7a2a">{data["grounded"]}/{data["total_questions"]}</div>
            <div class="bar"><div class="bar-fill" style="width:{grounded_pct}%;background:#2a7a2a"></div></div>
            <div class="card-sub">{grounded_pct}% of answers</div>
        </div>
        <div class="card">
            <div class="card-label">Hallucinated</div>
            <div class="card-value" style="color:{'#cc0000' if data['hallucinated'] > 0 else '#2a7a2a'}">{data["hallucinated"]}</div>
            <div class="card-sub">{'Needs attention' if data['hallucinated'] > 0 else 'None detected'}</div>
        </div>
        <div class="card">
            <div class="card-label">Avg grounding score</div>
            <div class="card-value">{data["avg_hallucination_score"]}</div>
            <div class="bar"><div class="bar-fill" style="width:{round(data['avg_hallucination_score']*100)}%"></div></div>
            <div class="card-sub">0.75+ is strong</div>
        </div>
        <div class="card">
            <div class="card-label">Avg retrieval score</div>
            <div class="card-value">{data["avg_retrieval_score"]}</div>
            <div class="bar"><div class="bar-fill" style="width:{retrieval_pct}%"></div></div>
            <div class="card-sub">0.4+ is good</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Question</th>
                <th>Answer (preview)</th>
                <th>Pages</th>
                <th>Hallucination</th>
                <th>Retrieval</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>

    <p style="font-size:12px;color:#bbb;text-align:center;margin-top:2rem">
        RAG Doc Intelligence — github.com/arya312/rag-doc-intelligence
    </p>
</body>
</html>"""

    with open("dashboard.html", "w") as f:
        f.write(html)
    print("Dashboard saved to dashboard.html")
    return html


if __name__ == "__main__":
    generate_dashboard()
    print("Serving dashboard at http://localhost:8080")
    print("Open the Ports tab and forward port 8080 to view it in browser")

    os.chdir(".")
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", 8080), handler) as httpd:
        httpd.serve_forever()
