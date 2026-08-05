import zipfile
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
keras_path = os.path.join(BASE_DIR, "app", "conv1_lstm.keras")

print("Inspecting and patching Keras model:", keras_path)

if not os.path.exists(keras_path):
    print("Error: conv1_lstm.keras not found!")
    exit(1)

with zipfile.ZipFile(keras_path, 'r') as zip_ref:
    config_bytes = zip_ref.read('config.json')
    config = json.loads(config_bytes.decode('utf-8'))

def fix_config(obj):
    patched_count = 0
    if isinstance(obj, dict):
        if obj.get('class_name') == 'BatchNormalization':
            cfg = obj.get('config', {})
            if isinstance(cfg.get('axis'), list) and len(cfg['axis']) == 1:
                cfg['axis'] = cfg['axis'][0]
                patched_count += 1
        elif obj.get('class_name') in ('LSTM', 'CuDNNLSTM'):
            cfg = obj.get('config', {})
            if 'time_major' in cfg:
                del cfg['time_major']
                patched_count += 1
        for v in obj.values():
            patched_count += fix_config(v)
    elif isinstance(obj, list):
        for item in obj:
            patched_count += fix_config(item)
    return patched_count

count = fix_config(config)
print("Patched layers count:", count)

temp_path = os.path.join(BASE_DIR, "app", "conv1_lstm_fixed.keras")
with zipfile.ZipFile(keras_path, 'r') as xin:
    with zipfile.ZipFile(temp_path, 'w') as xout:
        for item in xin.infolist():
            data = xin.read(item.filename)
            if item.filename == 'config.json':
                data = json.dumps(config, indent=2).encode('utf-8')
            xout.writestr(item, data)

os.replace(temp_path, keras_path)
print("Successfully patched conv1_lstm.keras for Keras 3 compatibility!")
