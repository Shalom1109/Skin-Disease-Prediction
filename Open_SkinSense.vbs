Set WshShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")
strFolder = objFSO.GetParentFolderName(WScript.ScriptFullName)

' Start the Python backend server completely silently (0 means hidden)
backendPath = Chr(34) & strFolder & "\backend" & Chr(34)
WshShell.Run "cmd.exe /c cd /d " & backendPath & " && python -m uvicorn app:app --host 127.0.0.1 --port 8000", 0, False

' Show a temporary popup so the user knows it is loading (closes automatically after 8 seconds)
WshShell.Popup "SkinSense AI is starting in the background. Please wait ~15 seconds for the heavy PyTorch AI model to load. The browser will open automatically when it is ready.", 8, "SkinSense AI Starting", 64

' Wait 15 seconds for the server to load the AI model
WScript.Sleep 15000

' Open the HTML file directly in the default browser
frontendPath = Chr(34) & strFolder & "\frontend\index.html" & Chr(34)
WshShell.Run frontendPath
