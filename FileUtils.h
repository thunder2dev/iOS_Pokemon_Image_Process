#pragma once


#include<string>
#include<vector>
#include<map>

using namespace std;

map<int, vector<string>> getFilesInFolder(char* filespec, char* dir, bool test = false);
