srcdir = .
bindir = bin
installdir = ~/.local/bin

VPATH = $(bindir):$(installdir)

FC = gfortran
FFLAGS = -O3 --std=legacy

targets = meangen stagen multall

.PHONY: all install prep clean

all: install

install: $(targets)
	chmod +x $(bindir)/*
	mkdir -p $(installdir)
	ln -sf $$PWD/$(bindir)/* $(installdir)

prep:
	mkdir -p $(bindir)

clean:
	rm -rf $(bindir)
	rm -rf $(installdir)

meangen: Meangen/meangen-17.4-custom.f prep
	$(FC) $(FFLAGS) -o $(bindir)/$@ $(srcdir)/$<

stagen: Stagen/stagen-18.1.f prep
	$(FC) $(FFLAGS) -o $(bindir)/$@ $(srcdir)/$<

multall: Multall/multall-open-18.2.f prep
	$(FC) $(FFLAGS) -o $(bindir)/$@ $(srcdir)/$<

