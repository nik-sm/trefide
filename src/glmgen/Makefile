CC=icc
CFLAGS=-O3 -Wall -Wextra -ansi -std=c89 -pedantic
CFLAGS2=-O3
OBJ=obj/*.o
IDIR=../include/

all:
	mkdir -p lib
	mkdir -p obj
	cd obj; ${CC} ${CLFAGS2} -c -fPIC ../src/csparse/*.c -I${IDIR}
	cd obj; ${CC} ${CFLAGS}  -c -fPIC ../src/utils/*.c -I${IDIR}
	cd obj; ${CC} ${CFLAGS}  -c -fPIC ../src/tf/*.c -I${IDIR}
	${CC} -shared -o lib/libglmgen.so ${OBJ}

clean:
	rm -rf lib
	rm -rf obj