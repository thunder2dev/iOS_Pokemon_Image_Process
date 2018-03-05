#include "FileUtils.h"
#include "stdlib.h"  
#include "direct.h"  
#include "string.h"  
#include "io.h"  
#include "stdio.h"   
#include "iostream"  


map<int, vector<string>> getFilesInFolder(char* filespec, char* dir, bool test) {

	map<int, vector<string>> paths;

	_chdir(dir);

	long hFile;
	_finddata_t fileinfo;
	if ((hFile = _findfirst(filespec, &fileinfo)) != -1)
	{
		do
		{
			if (!(fileinfo.attrib & _A_SUBDIR))
			{
				char filename[_MAX_PATH];
				strcpy(filename, dir);
				strcat(filename, fileinfo.name);
				
				int kind, num;

				if (test) {
					sscanf(fileinfo.name, "%d-test-%d.png", &kind, &num);
				}
				else {
					sscanf(fileinfo.name, "%d-%d.png", &kind, &num);
				}

				if (paths.find(kind) == paths.end()) {
					paths[kind] = vector<string>();
				}

				paths[kind].push_back(filename);

			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}

	return paths;


}