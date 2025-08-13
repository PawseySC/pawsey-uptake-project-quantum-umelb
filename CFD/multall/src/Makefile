srcdir = .
bindir = bin
installdir = ~/.local/bin

VPATH = $(bindir):$(installdir)

FC = gfortran
FFLAGS = -O3 --std=legacy

targets = meangen stagen multall

.PHONY: all install installdirs clean

all: $(targets)

install:
	mkdir -p $(installdir)
	ln -sf $$PWD/$(bindir)/* $(installdir)

installdirs:
	mkdir -p $(bindir)

clean:
	-rm -r $(bindir)
	-rm $(foreach target,$(targets),$(installdir)/$(target))

meangen: Meangen/meangen-17.4-custom.f installdirs
	$(FC) $(FFLAGS) -o $(bindir)/$@ $(srcdir)/$<
	chmod +x $(bindir)/$@

stagen: Stagen/stagen-18.1.f installdirs
	$(FC) $(FFLAGS) -o $(bindir)/$@ $(srcdir)/$<
	chmod +x $(bindir)/$@

multall: Multall/multall-open-18.2.f installdirs
	$(FC) $(FFLAGS) -o $(bindir)/$@ $(srcdir)/$<
	chmod +x $(bindir)/$@

