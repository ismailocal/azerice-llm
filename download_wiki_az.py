import requests
import json
import time

output_file = "wiki_az_api.jsonl"
base_url = "https://datasets-server.huggingface.co/rows"
params = {
    "dataset": "wikimedia/wikipedia",
    "config": "20231101.az",
    "split": "train",
    "offset": 0,
    "length": 100
}

with open(output_file, "w", encoding="utf-8") as f:
    while True:
        print(f"Çekiliyor: offset={params['offset']}")
        resp = requests.get(base_url, params=params)
        if resp.status_code != 200:
            print("Hata veya veri bitti.")
            break
        data = resp.json()
        rows = data.get("rows", [])
        if not rows:
            break
        for row in rows:
            text = row["row"].get("text")
            if text and 100 < len(text) < 1000:
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write("\n")
        params["offset"] += params["length"]
        time.sleep(0.5)  # API'yı yormamak için kısa bekleme

print("İşlem tamamlandı.") 