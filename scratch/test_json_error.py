import json

# Test 1: trailing comma after action
s1 = '{\n  "action": "ALLOW",\n'
try:
    json.loads(s1)
except Exception as e:
    print("s1 error:", e)

# Test 2: unterminated string
s2 = '{\n  "action": "ALLOW'
try:
    json.loads(s2)
except Exception as e:
    print("s2 error:", e)

# Test 3: cut off at line 2 char 20
s3 = '{\n"action": "ALLOW",\n'
try:
    json.loads(s3)
except Exception as e:
    print("s3 error:", e)

