# Check file
path = r'c:\Avicena\traffict\competition\big-data-traffict-competitiom\app\templates\settings.html'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
print('Size:', len(content), 'bytes')
checks = [
    ('{% extends', '{% extends' in content),
    ('{% block style_extra', '{% block style_extra' in content),
    ('{% endblock %}', '{% endblock %}' in content),
    ('gpu-bar', 'gpu-bar' in content),
    ('toast-bar', 'toast-bar' in content),
    ('confirm-overlay', 'confirm-overlay' in content),
    ('openConfirm', 'openConfirm' in content),
    ('doDelete', 'doDelete' in content),
    ('showToast', 'showToast' in content),
    ('tab-agent-tools', 'tab-agent-tools' in content),
    ('tab-hikvision', 'tab-hikvision' in content),
    ('tab-social', 'tab-social' in content),
    ('tab-enforcement', 'tab-enforcement' in content),
    ('btn-test-x', 'btn-test-x' in content),
    ('loadAgentTools', 'loadAgentTools' in content),
    ('loadViaStatus', 'loadViaStatus' in content),
    ('tab-ai-provider', 'tab-ai-provider' in content),
    ('tab-detection', 'tab-detection' in content),
    ('toggleAiMode', 'toggleAiMode' in content),
    ('btn-restart', 'btn-restart' in content),
    ('escHtml', 'escHtml' in content),
    ('escJs', 'escJs' in content),
]
for name, result in checks:
    status = 'OK' if result else 'MISSING'
    print(status, '-', name)
print('Ends with endblock:', content.strip().endswith('{% endblock %}'))
