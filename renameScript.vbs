Set objFso = CreateObject("Scripting.FileSystemObject")
Set Folder = objFSO.GetFolder("C:\Python\Sound_Computing\HeartBeats\heartbeat-sounds\set_b")

For Each File In Folder.Files
    sNewFile = File.Name
    sNewFile = Replace(sNewFile,"__","_")
    if (sNewFile<>File.Name) then 
        File.Move(File.ParentFolder+"\"+sNewFile)
    end if

Next