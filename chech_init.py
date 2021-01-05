from modeling import rc, spectrum

fetch = 3000


rc.surface.nonDimWindFetch = fetch
# k = spectrum.k0
# S = spectrum.spectrum(k)
spectrum.cov()