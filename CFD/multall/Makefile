srcdir = .
bindir = bin
installdir = ~/.local/bin

VPATH = $(bindir):$(installdir)

FC = gfortran
FFLAGS = -O3 --std=legacy

targets = bin/meangen bin/stagen bin/multall

.PHONY: all clean
# .PHONY: all clean meangen stagen multall

all: $(targets)

clean:
	rm -rf bin/*

bin/meangen: src/Meangen/meangen-17.4-custom.f
	$(FC) $(FFLAGS) -o $@ $<
bin/stagen: src/Stagen/stagen-18.1.f
	$(FC) $(FFLAGS) -o $@ $(srcdir)/$<
bin/multall: src/Multall/multall-open-18.2.f
	$(FC) $(FFLAGS) -o $@ $(srcdir)/$<

