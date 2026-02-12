#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void mysleep()
{
  sleep(30);
}
void myfun()
{
  mysleep();
}
int main(int argc, char **argv)
{
  myfun();
  return 0;
}
