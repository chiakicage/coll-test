BUILDDIR ?= build
override BUILDDIR := $(abspath $(BUILDDIR))

.PHONY : all clean

default : src.build

TARGETS=src

all:   ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

%.clean:
	${MAKE} -C $* clean BUILDDIR=${BUILDDIR}
