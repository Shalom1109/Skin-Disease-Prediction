Set WshShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")
strFolder = objFSO.GetParentFolderName(WScript.ScriptFullName)

' Start the Python backend server silently (0 = hidden window)
backendPath = Chr(34) & strFolder & "\backend" & Chr(34)
WshShell.Run "cmd.exe /c cd /d " & backendPath & " && python -m uvicorn app:app --host 127.0.0.1 --port 8000", 0, False

WshShell.Popup "SkinSense AI is starting in the background. The PyTorch model takes ~15 seconds to load; the browser opens automatically.", 8, "SkinSense AI Starting", 64

' Give the model time to load
WScript.Sleep 15000

' Open through the server (http://), NOT as a file:// page, so CORS can stay locked down
WshShell.Run "http://127.0.0.1:8000"
