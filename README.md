# HDD — Haleakala Disambiguation Decoder

**Project:** SPIn4D (Spectropolarimetric Inversion in Four Dimensions)  
**Paper:** *SPIn4D II: A Physics-Informed Machine Learning Method for 3D Solar Photosphere Reconstruction* (submitted to ApJ)

---

## Overview
HDD is a **physics-informed machine learning** method to reconstruct the **3D structure of the lower solar atmosphere** from **optical-depth–sampled spectropolarimetric inversions**. HDD jointly:
- **Maps optical depth ($\tau$) --> geometric height (z)**, and  
- **Enforces $\nabla\cdot B = 0$ in full 3D** while returning **fully disambiguated vector magnetic fields** together with the geometric height associated with each optical depth.

This repository contains the HDD codebase developed under the **SPIn4D** project.

---

## Highlights
- **Joint geometry + field inference:** Simultaneous recovery of geometric heights and azimuth-disambiguated magnetic fields.  
- **Hard physical consistency:** Divergence-free magnetic fields enforced in 3D.  
- **Designed for inversion outputs:** Operates directly on optical-depth–sampled inversion results.

---

## Status
- **Version:** v0.0.0 (2023-11-17)  
- **Code & data:** Will be uploaded soon. This README will be updated with installation, usage, and examples upon release.

---

## Citation
If you use HDD or ideas from this work, please cite the associated SPIn4D paper once available:

> *Spectropolarimetric Inversion in Four Dimensions (SPIn4D): II. A Physics-Informed Machine Learning Method for 3D Solar Photosphere Reconstruction, submitted to ApJ.*

A BibTeX entry will be provided after publication.

---

## Contact
**Kai Yang** — <yangkai@hawaii.edu>

---

## Acknowledgments
Part of the **SPIn4D** project. We thank collaborators and the broader community for discussions and feedback.

---

## License
TBD
