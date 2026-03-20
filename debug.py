import traceback
try:
    exec(open('app.py', encoding='utf-8').read())
except Exception as e:
    traceback.print_exc()
    input("Press Enter to exit...")
