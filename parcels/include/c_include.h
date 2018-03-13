#include <stdio.h>

typedef struct
{ 
  void *name;      
  int xi, yi;
} CGridIndex;        

typedef struct
{ 
  int size;
  CGridIndex *grids;
} CGridIndexSet;        


int printFunc()
{
  printf("printed from the included .h file \n");
  return 5;
}  

