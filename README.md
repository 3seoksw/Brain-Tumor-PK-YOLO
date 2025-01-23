# Official PK-YOLO
This is the source code for the paper titled "PK-YOLO: Pretrained Knowledge Guided YOLO for Brain Tumor Detection in Multiplane MRI Slices" accepted by the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision ([WACV 2025](https://wacv2025.thecvf.com)), of which I am the first author. The paper is available to download from [arXiv](https://arxiv.org/pdf/2410.21822). 
<!--
submitted to WACV 2025 (Paper ID: 466). This repository will be private before final decisions released to authors, i.e., Oct 28th, 2024.
## Errata
'Mamaba YOLO', which is a typo in the first version of the manuscript, shoule be Mamba YOLO.
-->
## Model

##### Installation
Install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.7.0.
```
pip install -r requirements.txt
```

## Referencing Guide
Please cite our paper if you use code from this repository. Here is a guide to referencing this work in various styles for formatting your references:
> Plain Text
- IEEE Full Name Reference Style</br>
Ming Kang, Fung Fung Ting, Raphaël C.-W. Phan, and Chee-Ming Ting. Pk-yolo: Pretrained knowledge guided yolo for brain tumor detection in multiplane mri slices. In *WACV*, in press, 2025.</br>
<sup>**NOTE:** This is a modification to the standard IEEE Reference Style and used by most IEEE/CVF conferences, including *CVPR*, *ICCV*, and *WACV*, to render first names in the bibliography as "Firstname Lastname" rather than "F. Lastname" or "Lastname, F.", which the reference styles of *NeurIPS*, *ICLR*, and *IJCAI* are similar to.</sup>

- IEEE Reference Style</br>
M. Kang, F. F. Ting, R. C.-W. Phan, and C.-M. Ting, "Pk-yolo: Pretrained knowledge guided yolo for brain tumor detection in multiplane mri slices," in *Proc. Winter Conf. Appl. Comput. Vis. (WACV)*, Tucson, AZ, USA, Feb. 28–Mar. 4, 2025, in press.</br>
<sup>**NOTE:** City of Conf., Abbrev. State, Country, Month & Day(s) are optional.</sup>

- Nature Reference Style</br>
Kang, M., Ting, C.-M., Ting, F. F. & Phan, R. C.-W. PK-YOLO: pretrained knowledge guided YOLO for brain tumor detection in multiplane MRI slices. In *2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)* in press (IEEE, 2025).</br>

- Springer Reference Style</br>
Kang, M., Ting, F.F., Phan, R.C.-W., Ting, C.-M.: PK-YOLO: pretrained knowledge guided YOLO for brain tumor detection in multiplane MRI slices. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), in press. IEEE, Piscataway (2025)</br>
<sup>**NOTE:** MICCAI conference proceedings are part of the book series LNCS in which Springer's format for bibliographical references is strictly enforced. This is important, for instance, when citing previous MICCAI proceedings. LNCS stands for Lecture Notes in Computer Science.</sup>

- Elsevier Numbered Style</br>
M. Kang, F.F. Ting, R.C.-W. Phan, C.-M. Ting, PK-YOLO: Pretrained knowledge guided YOLO for brain tumor detection in multiplane MRI slices, in: Proceedings of the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 8 Februray–4 March 2025, Tucson, AZ, USA. IEEE, Piscataway, New York, USA, in press.</br>
<sup>**NOTE:** Day(s) Month Year, City, Abbrev. State, Country of Conference, Publiser, and Place of Publication are optional.</sup>

- Harvard (Name–Date) Style</br>
Kang, M., Ting, F.F., Phan, R.C.-W., Ting, C.-M., 2025. PK-YOLO: Pretrained knowledge guided YOLO for brain tumor detection in multiplane MRI slice. In: Proceedings of the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 28 Februray–4 March 2025, Tucson, AZ, USA. IEEE, Piscataway, New York, USA, in press.</br>
<sup>**NOTE:** Day(s) Month Year, City, Abbrev. State, Country of Conference, Publiser, and Place of Publication are optional.</sup>

- APA7 (Author–Date) Style</br>
Kang, M., Ting, F.F., Phan, R.C.-W., & Ting, C.-M. (2025). PK-YOLO: Pretrained knowledge guided YOLO for brain tumor detection in multiplane MRI slice. In *Proceedings of the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)* (in press). IEEE. https://doi.org/10.1109/WACV56688.2025.00000</br>


> BibTeX Format</br>
```
\begin{thebibliography}{1}
\bibitem{Kang25Pkyolo} M. Kang, F. F. Ting, R. C.-W. Phan, and C.-M. Ting, "Pk-yolo: Pretrained knowledge guided yolo for brain tumor detection in multiplane mri slices," in {\emph Proc. Winter Conf. Appl. Comput. Vis. (WACV)}, Tucson, AZ, USA, Feb. 28–Mar. 4, 2025, in press.
\end{thebibliography}
```
```
@inproceedings{Kang25Pkyolo,
  author = "Ming Kang and Fung Fung Ting and Rapha{\"e}l C.-W. Phan and Chee-Ming Ting",
  title = "Pk-yolo: Pretrained knowledge guided yolo for brain tumor detection in multiplane mri slices",
  booktitle = "Proc. Winter Conf. Appl. Comput. Vis. (WACV)",
  address = "Tucson, AZ, USA, Feb. 28--Mar. 4",
  pages = "in press",
  year = "2025"
}
```
```
@inproceedings{Kang25Pkyolo,
  author = "Kang, Ming and Ting, Fung Fung and Phan, Rapha{\"e}l C.-W. and Ting, Chee-Ming",
  title = "{PK-YOLO}: pretrained knowledge guided {YOLO} for brain tumor detection in multiplane {MRI} slices",
  editor = "",
  booktitle = "2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)",
  series = "",
  volume = "",
  pages = "in press",
  publisher = "IEEE",
  address = "Piscataway",
  year = "2025",
  doi= "10.1109/WACV56688.2025.00000",
  url = "https://doi.org/10.1109/WACV56688.2025.00000"
}
```
<sup>**NOTE:** Please remove some optional *BibTeX* fields/tags such as `series`, `volume`, `address`, `url`, and so on if the *LaTeX* compiler produces an error. Author names may be manually modified if not automatically abbreviated by the compiler under the control of the bibliography/reference style (i.e., .bst) file. The *BibTex* citation key may be `bib1`, `b1`, or `ref1` when references appear in numbered style in which they are cited. The quotation mark pair `""` in the field could be replaced by the brace `{}`, whereas the brace `{}` in the *BibTeX* field/tag `title` plays a role of keeping letters/characters/text original lower/uppercases or sentence/capitalized cases unchanged while using Springer Nature bibliography style files, for example, sn-nature.bst.</sup>

## License
PK-YOLO is released under the GNU General Public License v3.0. Please see the [LICENSE](https://github.com/mkang315/PK-YOLO/blob/main/LICENSE) file for more information.

## Copyright Notice
Many utility codes of our project base on the codes of [YOLOv9](https://github.com/WongKinYiu/yolov9) and [SparK](https://github.com/keyu-tian/SparK) repositories.
