"""Replace 'key' references in Step 3 of enforcement.py with 'tk'."""
import re

path = r'c:\Avicena\traffict\competition\big-data-traffict-competitiom\app\services\enforcement.py'
content = open(path, encoding='utf-8').read()

# Find the Step 3 section
marker = 'for tk, st in list(self._tracked.items()):'
idx = content.find(marker)
if idx < 0:
    print("Marker not found!")
    exit(1)

before = content[:idx]
after = content[idx:]

# Replace key references in the after section
replacements = [
    ('key[0]', 'tk[0]'),
    ('f"_anpr_fb_{key}"', 'f"_anpr_fb_{tk}"'),
    ('f"_vid_running_{key}"', 'f"_vid_running_{tk}"'),
    ('_fb_key = f"_anpr_fb_{key}"', '_fb_key = f"_anpr_fb_{tk}"'),
    ('_vk = f"_vid_running_{key}"', '_vk = f"_vid_running_{tk}"'),
    ('[ANPR-SYNC] Track {key}', '[ANPR-SYNC] Track {tk}'),
    ('seed=f"{self.camera_id}:{key}:sync"', 'seed=f"{self.camera_id}:{tk}:sync"'),
    ('str(key), self.camera_name)', 'str(tk), self.camera_name)'),
    ('str(key))', 'str(tk))'),
    (', key), daemon', ', tk), daemon'),
]

for old, new in replacements:
    count = after.count(old)
    if count > 0:
        after = after.replace(old, new)
        print(f"  Replaced '{old}' -> '{new}' ({count}x)")

content = before + after
open(path, 'w', encoding='utf-8').write(content)
print("\nDone!")
