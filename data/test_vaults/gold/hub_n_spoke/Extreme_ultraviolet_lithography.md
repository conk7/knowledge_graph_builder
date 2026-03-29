Extreme ultraviolet lithography (EUVL or simply EUV) is a technology used in the semiconductor industry for manufacturing integrated circuits (ICs). It is a type of photolithography that uses 13.5 nm extreme ultraviolet (EUV) light from a laser-pulsed tin (Sn) plasma to create intricate patterns on semiconductor substrates. 
As of 2025, ASML Holding is the only company that produces and sells EUV systems for chip production, targeting 5 nanometer (nm) and 3 nm process nodes, though Reuters reported in December 2025 that China had developed its own prototype EUV system.
The EUV wavelengths that are used in EUVL are near 13.5 nanometers (nm), using a laser-pulsed tin (Sn) droplet plasma to produce a pattern by using a reflective photomask to expose a substrate covered by photoresist.  Tin ions in the ionic states from Sn IX to Sn XIV give photon emission spectral peaks around 13.5 nm from 4p64dn – 4p54dn+1 + 4dn−14f ionic state transitions.

History and economic impact
In the 1960s, visible light was used for the production of integrated circuits, with wavelengths as short as 435 nm (mercury "g line").
Later, ultraviolet (UV) light was used, at first with a wavelength of 365 nm (mercury "i line"), then with excimer wavelengths, first of 248 nm (krypton fluoride laser), then 193 nm (argon fluoride laser), which was called deep UV.
The next step, going even smaller, was called extreme UV, or EUV. The EUV technology was considered impossible by many.
EUV light is absorbed by glass and air, so instead of using lenses to focus the beams of light as done previously, mirrors in vacuum would be needed. A reliable production of EUV was also problematic. Then, leading producers of steppers Canon and Nikon stopped development, and some predicted the end of Moore's law.
While working at Nippon Telegraph and Telephone (NTT) in mid-1980s Japan, engineer Hiroo Kinoshita first proposed the concept of EUV. He tested the idea and successfully demonstrated the first EUV images at a 1986 Japan Society of Applied Physics (JSAP) meeting. Despite initial scepticism in Japan, Kinoshita continued EUV research at NTT and organized joint US-Japan research on EUV in the early 1990s.
In 1991, scientists at Bell Labs published a paper demonstrating the possibility of using a wavelength of 13.8 nm for the so-called soft X-ray projection lithography.
To address the challenge of EUV lithography, researchers at Lawrence Livermore National Laboratory, Lawrence Berkeley National Laboratory, and Sandia National Laboratories were funded in the 1990s to perform basic research into the technical obstacles. The results of this successful effort were disseminated via a public/private partnership Cooperative R&D Agreement (CRADA). The CRADA consisted of a consortium of private companies and the Labs, manifested as an entity called the Extreme Ultraviolet Limited Liability Company (EUV LLC). Meanwhile back in Japan, EUV technology development was pursued in the 1990s through the ASET (Association of Super-Advanced Electronics Technologies) and Extreme Ultraviolet Lithography Development Association (EUVA) programs.
Intel, Canon, and Nikon (leaders in the field at the time), as well as the Dutch company ASML and Silicon Valley Group (SVG) all sought licensing. In 2001, SVG was acquired by ASML, helping ASML become the leading benefactor of the critical technology.
By 2018, ASML succeeded in deploying the intellectual property from the EUV-LLC after several decades of developmental research, with incorporation of European-funded EUCLIDES (Extreme UV Concept Lithography Development System) and long-standing partner German optics manufacturer ZEISS and synchrotron light source supplier Oxford Instruments. This led MIT Technology Review to name it "the machine that saved Moore's law". Their first prototype in 2006 produced one wafer in 23 hours. As of 2022, a scanner produces up to 200 wafers per hour. The scanner uses Zeiss optics, which that company calls "the most precise mirrors in the world", produced by locating imperfections and then knocking off individual molecules with techniques such as ion beam figuring.
This made the once small company ASML the world leader in the production of scanners and monopolist in this cutting-edge technology and resulted in a record turnover of 27.4 billion euros in 2021, dwarfing their competitors Canon and Nikon, who were denied IP access. Because it is such a key technology for development in many fields, the United States licenser pressured Dutch authorities to not sell these machines to China. ASML has followed the guidelines of Dutch export controls and until further notice will have no authority to ship the machines to China. China, at the same time, also has invested heavily into their domestic EUV project, and Chinese leading companies such as Huawei and SMEE also filed patents for their alternative proposals relevant to EUV technologies. In December 2025, Reuters reported that China had secretly completed a prototype EUV machine in Shenzhen, which was expected to produce working chips between 2028 and 2030.
Along with multiple patterning, EUV has paved the way for higher transistor densities, allowing the production of higher-performance processors. Smaller transistors also require less power to operate, resulting in more energy-efficient electronics.

Fab tool output
Requirements for EUV steppers, given the number of layers in the design that require EUV, the number of machines, and the desired throughput of the fab, assuming 24 hours per day operation.

Masks
EUV photomasks work by reflecting light, which is achieved by using multiple alternating layers of molybdenum and silicon. This is in contrast to conventional photomasks which work by blocking light using a single chromium layer on a quartz substrate. An EUV mask consists of 40–50 alternating silicon and molybdenum layers; this is a multilayer which acts to reflect the extreme ultraviolet light through Bragg diffraction; the reflectance is a strong function of incident angle and wavelength, with longer wavelengths reflecting more near normal incidence and shorter wavelengths reflecting more away from normal incidence. The multilayer may be protected by a thin ruthenium layer, called a capping layer. The pattern is defined in a tantalum-based absorbing layer over the capping layer. 
Blank photomasks are mainly made by two companies: AGC Inc. and Hoya Corporation. Ion-beam deposition equipment mainly made by Veeco is often used to deposit the multilayer. A blank photomask is covered with photoresist, which is then baked (solidified) in an oven, and later the pattern is defined on the photoresist using maskless lithography with an electron beam. This step is called exposure. The exposed photoresist is developed (removed), and the unprotected areas are etched. The remaining photoresist is then removed. Masks are then inspected and later repaired using an electron beam. Etching must be done only in the absorbing layer and thus there is a need to distinguish between the capping and the absorbing layer, which is known as etch selectivity and is unlike etching in conventional photomasks, which only have one layer critical to their function.

Tool
An EUV tool (EUV photolithography machine) has a laser-driven tin (Sn) plasma light source, reflective optics comprising multilayer mirrors, contained within a hydrogen gas ambient. The hydrogen is used to keep the EUV collector mirror, as the first mirror collecting EUV emitted over a large range in angle (~2π sr) from the Sn plasma, in the source free of Sn deposition. Specifically, the hydrogen buffer gas in the EUV source chamber or vessel decelerates or possibly pushes back Sn ions and Sn debris traveling toward the EUV collector (collector protection) and enable a chemical reaction of 
  
    
      
        
          Sn
          
            (
            s
            )
          
          +
          4
          
          H
          
            (
            g
            )
          
          ⟶
          
            SnH
            
              4
            
            
              
            
          
          
            (
            g
            )
          
        
      
    
    {\displaystyle {\ce {Sn(s) + 4H(g) -> SnH4(g)}}}
  
 to remove Sn deposition on the collector in the form of 
  
    
      
        
          
            SnH
            
              4
            
            
              
            
          
        
      
    
    {\displaystyle {\ce {SnH4}}}
  
 gas (collector reflectivity restoration).
EUVL is a significant departure from the deep-ultraviolet lithography standard. All matter absorbs EUV radiation. Hence, EUV lithography requires vacuum. All optical elements, including the photomask, must use defect-free molybdenum/silicon (Mo/Si) multilayers (consisting of 50 Mo/Si bilayers, which have a theoretical reflectivity limit at 13.5 nm of ~75%) that act to reflect light by means of interlayer wave interference; any one of these mirrors absorb around 30% of the incident light, so the mirror temperature control is important.
EUVL systems, as of 2002-2009, contain at least two condenser multilayer mirrors, six projection multilayer mirrors and a multilayer object (mask). Since the mirrors absorb 96% of the EUV light, the ideal EUV source needs to be much brighter than its predecessors. EUV source development has focused on plasmas generated by laser or discharge pulses. The mirror responsible for collecting the light is directly exposed to the plasma and is vulnerable to damage from high-energy ions and other debris such as tin droplets, which require the costly collector mirror to be replaced every year.

Resource requirements
The required utility resources are significantly larger for EUV compared to 193 nm immersion, even with two exposures using the latter. At the 2009 EUV Symposium, Hynix reported that the wall plug efficiency was ~0.02% for EUV, i.e., to get 200 watts at intermediate focus for 100 wafers per hour, one would require 1 megawatt of input power, compared to 165 kilowatts for an ArF immersion scanner, and that even at the same throughput, the footprint of the EUV scanner was ~3× the footprint of an ArF immersion scanner, resulting in productivity loss. Additionally, to confine ion debris, a superconducting magnet may be required.
A typical EUV tool weighs nearly 200 tons and costs around 180 million USD.
EUV tools consume at least 10× more energy than immersion tools.

Summary of key features
The following table summarizes key differences between EUV systems in development and ArF immersion systems which are widely used in production today:

The different degrees of resolution among the 0.33 NA tools are due to the different illumination options. Despite the potential of the optics to reach sub-20 nm resolution, secondary electrons in resist practically limit the resolution to around 20 nm (more on this below).

Light source power, throughput, and uptime
Neutral atoms or condensed matter cannot emit EUV radiation. Ionization must precede EUV emission in matter. The thermal production of multicharged positive ions is only possible in a hot dense plasma, which itself strongly absorbs EUV. As of 2025, the established EUV light source is a laser-pulsed tin plasma. The ions absorb the EUV light they emit and are easily neutralized by electrons in the plasma to lower charge states, which produce light mainly at other, unusable wavelengths, resulting in a much reduced efficiency of light generation for lithography at higher plasma power density.
The throughput is tied to the source power, divided by the dose. A higher dose requires a slower stage motion (lower throughput) if pulse power cannot be increased.
EUV collector reflectivity degrades ~0.1–0.3% per billion 50 kHz pulses (~10% in ~2 weeks), leading to loss of uptime and throughput, while even for the first few billion pulses (within one day), there is still 20% (±10%) fluctuation. This could be due to the accumulating Sn residue mentioned above which is not completely cleaned off. On the other hand, conventional immersion lithography tools for double-patterning provide consistent output for up to a year.
Recently, the NXE:3400B illuminator features a smaller pupil fill ratio (PFR) down to 20% without transmission loss. PFR is maximized and greater than 0.2 around a metal pitch of 45 nm.
Due to the use of EUV mirrors which also absorb EUV light, only a small fraction of the source light is finally available at the wafer. There are 4 mirrors used for the illumination optics and 6 mirrors for the projection optics. The EUV mask or reticle is itself an additional mirror. With 11 reflections, only ~2% of the EUV source light is available at the wafer.
The throughput is determined by the EUV resist dose, which in turn depends on the required resolution. A dose of 40 mJ/cm2 is expected to be maintained for adequate throughput.

Tool uptime
The EUV light source limits tool uptime besides throughput. In a two-week period, for example, over seven hours downtime may be scheduled, while total actual downtime including unscheduled issues could easily exceed a day. A dose error over 2% warrants tool downtime.
The wafer exposure throughput steadily expanded up to around 1000 wafers per day (per system) over the 2019–2022 period, indicating substantial idle time, while at the same time running >120 wafers per day on a number of multipatterned EUV layers, for an EUV wafer on average.

Comparison to other lithography light sources
EUV (10–121 nm) is the band longer than X-rays (0.1–10 nm) and shorter than the hydrogen Lyman-alpha line (121 nm).
While state-of-the-art 193 nm ArF excimer lasers offer intensities of 200 W/cm2, lasers for producing EUV-generating plasmas need to be much more intense, on the order of 1011 W/cm2. A state-of-the-art ArF immersion lithography 120 W light source requires no more than 40 kW electrical power, while EUV sources are targeted to exceed 40 kW.
The optical power target for EUV lithography is at least 250 W, while for other conventional lithography sources, it is much less. For example, immersion lithography light sources target 90 W, dry ArF sources 45 W, and KrF sources 40 W. High-NA EUV sources are expected to require at least 500 W.

EUV-specific optical issues
Reflective optics
A fundamental aspect of EUVL tools, resulting from the use of reflective optics, is the off-axis illumination (at an angle of 6°, in different direction at different positions within the illumination slit) on a multilayer mask (reticle). This leads to shadowing effects resulting in asymmetry in the diffraction pattern that degrade pattern fidelity in various ways as described below. For example, one side (behind the shadow) would appear brighter than the other (within the shadow).
The behavior of light rays within the plane of reflection (affecting horizontal lines) is different from the behavior of light rays out of the plane of reflection (affecting vertical lines). Most conspicuously, identically sized horizontal and vertical lines on the EUV mask are printed at different sizes on the wafer.

The combination of the off-axis asymmetry and the mask shadowing effect leads to a fundamental inability of two identical features even in close proximity to be in focus simultaneously. One of EUVL's key issues is the asymmetry between the top and bottom line of a pair of horizontal lines (the so-called "two-bar"). Some ways to partly compensate are the use of assist features as well as asymmetric illumination.
An extension of the two-bar case to a grating consisting of many horizontal lines shows similar sensitivity to defocus. It is manifest in the critical dimension (CD) difference between the top and bottom edge lines of the set of 11 horizontal lines.
Polarization by reflection also leads to partial polarization of EUV light, which favors imaging of lines perpendicular to the plane of the reflections.

Pattern shift from defocus (non-telecentricity)
The EUV mask absorber, due to partial transmission, generates a phase difference between the 0th and 1st diffraction orders of a line-space pattern, resulting in image shifts (at a given illumination angle) as well as changes in peak intensity (leading to linewidth changes) which are further enhanced due to defocus. Ultimately, this results in different positions of best focus for different pitches and different illumination angles. Generally, the image shift is balanced out due to illumination source points being paired (each on opposite sides of the optical axis). However, the separate images are superposed and the resulting image contrast is degraded when the individual source image shifts are large enough. The phase difference ultimately also determines the best focus position.
The multilayer is also responsible for image shifting due to phase shifts from diffracted light within the multilayer itself. This is inevitable due to light passing twice through the mask pattern.
The use of reflection causes wafer exposure position to be extremely sensitive to the reticle flatness and the reticle clamp. Reticle clamp cleanliness is therefore required to be maintained. Small (milliradian-scale) deviations in mask flatness in the local slope, coupled with wafer defocus. More significantly, mask defocus has been found to result in large overlay errors. In particular, for a 10 nm node metal 1 layer (including 48 nm, 64 nm, 70 nm pitches, isolated, and power lines), the uncorrectable pattern placement error was 1 nm for 40 nm mask z-position shift. This is a global pattern shift of the layer with respect to previously defined layers. However, features at different locations will also shift differently due to different local deviations from mask flatness, e.g., from defects buried under the multilayer. It can be estimated that the contribution of mask non-flatness to overlay error is roughly 1/40 times the peak-to-valley thickness variation. With the blank peak-to-valley spec of 50 nm, ~1.25 nm image placement error is possible. Blank thickness variations up to 80 nm also contribute, which lead to up to 2 nm image shift.
The off-axis illumination of the reticle is also the cause of non-telecentricity in wafer defocus, which consumes most of the 1.4 nm overlay budget of the NXE:3400 EUV scanner even for design rules as loose as 100 nm pitch. The worst uncorrectable pattern placement error for a 24 nm line was about 1.1 nm, relative to an adjacent 72 nm power line, per 80 nm wafer focus position shift at a single slit position; when across-slit performance is included, the worst error is over 1.5 nm in the wafer defocus window In 2017, an actinic microscope mimicking a 0.33 NA EUV lithography system with 0.2/0.9 quasar 45 illumination showed that an 80 nm pitch contact array shifted −0.6 to 1.0 nm while a 56 nm pitch contact array shifted −1.7 to 1.0 nm relative to a horizontal reference line, within a ±50 nm defocus window.
Wafer defocus also leads to image placement errors due to deviations from local mask flatness. If the local slope is indicated by an angle α, the image is projected to be shifted in a 4× projection tool by 8α × (DOF/2) = 4α DOF, where DOF is the depth of focus. For a depth of focus of 100 nm, a small local deviation from flatness of 2.5 mrad (0.14°) can lead to a pattern shift of 1 nm.
Simulations as well as experiments have shown that pupil imbalances in EUV lithography can result in pitch-dependent pattern placement errors. Since the pupil imbalance changes with EUV collector mirror aging or contamination, such placement errors may not be stable over time. The situation is specifically challenging for logic devices, where multiple pitches have critical requirements at the same time. The issue is ideally addressed by multiple exposures with tailored illuminations.

Slit position dependence
The direction of illumination is also highly dependent on slit position, essentially rotated azimuthally. Nanya Technology and Synopsys found that horizontal vs. vertical bias changed across slit with dipole illumination. The rotating plane of incidence (azimuthal range within −25° to 25°) is confirmed in the SHARP actinic review microscope at CXRO which mimics the optics for EUV projection lithography systems. The reason for this is that a mirror is used to transform straight rectangular fields into arc-shaped fields. In order to preserve a fixed plane of incidence, the reflection from the previous mirror would be from a different angle with the surface for a different slit position; this causes non-uniformity of reflectivity. To preserve uniformity, rotational symmetry with a rotating plane of incidence is used. More generally, so-called "ring-field" systems reduce aberrations by relying on the rotational symmetry of an arc-shaped field derived from an off-axis annulus. This is preferred, as reflective systems must use off-axis paths, which aggravate aberrations. Hence identical die patterns within different halves of the arc-shaped slit would require different OPC. This renders them uninspectable by die-to-die comparison, as they are no longer truly identical dies. For pitches requiring dipole, quadrupole, or hexapole illumination, the rotation also causes mismatch with the same pattern layout at a different slit position, i.e., edge vs. center. Even with annular or circular illumination, the rotational symmetry is destroyed by the angle-dependent multilayer reflectance described above. Although the azimuthal angle range is about ±20° (field data indicated over 18°) on 0.33 NA scanners, at 7 nm design rules (36–40 nm pitch), the tolerance for illumination can be ±15°, or even less. Annular illumination nonuniformity and asymmetry also significantly impact the imaging. Newer systems have azimuthal angle ranges going up to ±30°. On 0.33 NA systems, 30 nm pitch and lower already suffer sufficient reduction of pupil fill to significantly affect throughput.
The larger incident angle for pitch-dependent dipole illumination trend across slit does not affect horizontal line shadowing so much, but vertical line shadowing does increase going from center to edge. In addition, higher-NA systems may offer limited relief from shadowing, as they target tight pitches.

The slit position dependence is particularly difficult for the tilted patterns encountered in DRAM. Besides the more complicated effects due to shadowing and pupil rotation, tilted edges are converted to stair shape, which may be distorted by OPC. In fact, the 32 nm pitch DRAM by EUV will lengthen up to at least 9F2 cell area, where F is the active area half-pitch (traditionally, it had been 6F2). With a 2-D self-aligned double-patterning active area cut, the cell area is still lower at 8.9F2.
Aberrations, originating from deviations of optical surfaces from subatomic (<0.1 nm) specifications as well as thermal deformations and possibly including polarized reflectance effects, are also dependent on slit position, as will be further discussed below, with regard to source-mask optimization (SMO). The thermally induced aberrations are expected to exhibit differences among different positions across the slit, corresponding to different field positions, as each position encounters different parts of the deformed mirrors. Ironically, the use of substrate materials with high thermal and mechanical stability make it more difficult to compensate wavefront errors.
In combination with the range of wavelengths, the rotated plane of incidence aggravates the already severe stochastic impact on EUV imaging.

Wavelength bandwidth (chromatic aberration)
Unlike deep ultraviolet (DUV) lithography sources, based on excimer lasers, EUV plasma sources produce light across a broad range of wavelengths roughly spanning a 2% FWHM bandwidth near 13.5 nm (13.36 – 13.65 nm at 50% power). EUV (10–121 nm) is the band longer than X-Rays (0.1–10 nm) and shorter than the hydrogen Lyman-alpha line.
Though the EUV spectrum is not completely monochromatic, nor even as spectrally pure as DUV laser sources, the working wavelength has generally been taken to be 13.5 nm. In actuality, the reflected power is distributed mostly in the 13.3-13.7 nm range. The bandwidth of EUV light reflected by a multilayer mirror used for EUV lithography is over +/-2% (>270 pm); the phase changes due to wavelength changes at a given illumination angle may be calculated
and compared to the aberration budget. Wavelength dependence of reflectance also affects the apodization, or illumination distribution across the pupil (for different angles); different wavelengths effectively 'see' different illuminations as they are reflected differently by the multilayer of the mask. This effective source illumination tilt can lead to large image shifts due to defocus. Conversely, the peak reflected wavelength varies across the pupil due to different incident angles. This is aggravated when the angles span a wide radius, e.g., annular illumination. The peak reflectance wavelength increases for smaller incident angles. Aperiodic multilayers have been proposed to reduce the sensitivity at the cost of lower reflectivity but are too sensitive to random fluctuations of layer thicknesses, such as from thickness control imprecision or interdiffusion. 
A narrower bandwidth would increase sensitivity to mask absorber and buffer thickness on the 1 nm scale.

Flare
Flare is the presence of background light originating from scattering off of surface features which are not resolved by the light. In EUV systems, this light can be EUV or out-of-band (OoB) light that is also produced by the EUV source. The OoB light adds the complication of affecting the resist exposure in ways other than accounted for by the EUV exposure. OoB light exposure may be alleviated by a layer coated above the resist, as well as 'black border' features on the EUV mask. However, the layer coating inevitably absorbs EUV light, and the black border adds EUV mask processing cost.

Line tip effects
A key challenge for EUV is the counter-scaling behavior of the line tip-to-tip (T2T) distance as half-pitch (hp) is scaled down. This is in part due to lower image contrast for the binary masks used in EUV lithography, which is not encountered with the use of phase shift masks in immersion lithography. The rounding of the corners of the line end leads to line end shortening, and this is worse for binary masks. The use of phase-shift masks in EUV lithography has been studied but encounters difficulties from phase control in thin layers as well as the bandwidth of the EUV light itself. More conventionally, optical proximity correction (OPC) is used to address the corner rounding and line-end shortening. In spite of this, it has been shown that the tip-to-tip resolution and the line tip printability are traded off against each other, being effectively CDs of opposite polarity.
In unidirectional metal layers, tip-to-tip spacing is one of the more severe issues for single exposure patterning. For the 40 nm pitch vertical lines, an 18 nm nominal tip-to-tip drawn gap resulted in an actual tip-to-tip distance of 29 nm with OPC, while for 32 nm pitch horizontal lines, the tip-to-tip distance with a 14 nm nominal gap went to 31 nm with OPC. These actual tip-to-tip distances define a lower limit of the half-pitch of the metal running in the direction perpendicular to the tip. In this case, the lower limit is around 30 nm. With further optimization of the illumination (discussed in the section on source-mask optimization), the lower limit can be further reduced to around 25 nm.
For larger pitches, where conventional illumination can be used, the line tip-to-tip distance is generally larger. For the 24 nm half-pitch lines, with a 20 nm nominally drawn gap, the distance was actually 45 nm, while for 32 nm half-pitch lines, the same nominal gap resulted in a tip-to-tip distance of 34 nm. With OPC, these become 39 nm and 28 nm for 24 nm half-pitch and 32 nm half-pitch, respectively.

Enhancement opportunities for EUV patterning
Assist features
Assist features are often used to help balance asymmetry from non-telecentricity at different slit positions, due to different illumination angles, starting at the 7 nm node, where the pitch is ~ 41 nm for a wavelength ~13.5 nm and NA=0.33, corresponding to k1 ~ 0.5. However, the asymmetry is reduced but not eliminated, since the assist features mainly enhance the highest spatial frequencies, whereas intermediate spatial frequencies, which also affect feature focus and position, are not much affected. The coupling between the primary image and the self images is too strong for the asymmetry to be eliminated by assist features; only asymmetric illumination can achieve this. Assist features may also get in the way of access to power/ground rails. Power rails are expected to be wider, which also limits the effectiveness of using assist features, by constraining the local pitch. Local pitches between 1× and 2× the minimum pitch forbid assist feature placement, as there is simply no room to preserve the local pitch symmetry. In fact, for the application to the two-bar asymmetry case, the optimum assist feature placement may be less than or exceed the two-bar pitch. Depending on the parameter to be optimized (process window area, depth of focus, exposure latitude), the optimum assist feature configuration can be very different, e.g., pitch between assist feature and bar being different from two-bar pitch, symmetric or asymmetric, etc..
At pitches smaller than 58 nm, there is a tradeoff between depth of focus enhancement and contrast loss by assist feature placement. Generally, there is still a focus-exposure tradeoff as the dose window is constrained by the need to have the assist features not print accidentally.
An additional concern comes from shot noise; sub-resolution assist features (SRAFs) cause the required dose to be lower, so as not to print the assist features accidentally. This results in fewer photons defining smaller features (see discussion in section on shot noise).
As SRAFs are smaller features than primary features and are not supposed to receive doses high enough to print, they are more susceptible to stochastic dose variations causing printing errors; this is particularly prohibitive for EUV, where phase-shift masks may need to be used.

Source-mask optimization
Due to the effects of non-telecentricity, standard illumination pupil shapes, such as disc or annular, are not sufficient to be used for feature sizes of ~20 nm or below (10 nm node and beyond). Instead certain parts of the pupil (often over 50%) must be asymmetrically excluded. The parts to be excluded depend on the pattern. In particular, the densest allowed lines need to be aligned along one direction and prefer a dipole shape. For this situation, double exposure lithography would be required for 2D patterns, due to the presence of both X- and Y-oriented patterns, each requiring its own 1D pattern mask and dipole orientation. There may be 200–400 illuminating points, each contributing its weight of the dose to balance the overall image through focus. Thus the shot noise effect (to be discussed later) critically affects the image position through focus, in a large population of features.
Double- or multiple-patterning would also be required if a pattern consists of sub-patterns which require significantly different optimized illuminations, due to different pitches, orientations, shapes, and sizes.

Impact of slit position and aberrations
Largely due to the slit shape, and the presence of residual aberrations, the effectiveness of SMO varies across slit position. At each slit position, there are different aberrations and different azimuthal angles of incidence leading to different shadowing. Consequently, there could be uncorrected variations across slit for aberration-sensitive features, which may not be obviously seen with regular line-space patterns. At each slit position, although optical proximity correction (OPC), including the assist features mentioned above, may also be applied to address the aberrations, they also feedback into the illumination specification, since the benefits differ for different illumination conditions. This would necessitate the use of different source-mask combinations at each slit position, i.e., multiple mask exposures per layer.
The above-mentioned chromatic aberrations, due to mask-induced apodization, also lead to inconsistent source-mask optimizations for different wavelengths.

Pitch-dependent focus windows
The best focus for a given feature size varies as a strong function of pitch, polarity, and orientation under a given illumination. At 36 nm pitch, horizontal and vertical darkfield features have more than 30 nm difference of focus. The 34 nm pitch and 48 nm pitch features have the largest difference of best focus regardless of feature type. In the 48–64 nm pitch range, the best focus position shifts roughly linearly as a function of pitch, by as much as 10–20 nm. For the 34–48 nm pitch range, the best focus position shifts roughly linearly in the opposite direction as a function of pitch. This can be correlated with the phase difference between the zero and first diffraction orders. Assist features, if they can fit within the pitch, were found not to reduce this tendency much, for a range of intermediate pitches, or even worsened it for the case of 18–27 nm and quasar illumination. 50 nm contact holes on 100 nm and 150 pitches had best focus positions separated by roughly 25 nm; smaller features are expected to be worse. Contact holes in the 48–100 nm pitch range showed a 37 nm best focus range. The best focus position vs. pitch is also dependent on resist. Critical layers often contain lines at one minimum pitch of one polarity, e.g., darkfield trenches, in one orientation, e.g., vertical, mixed with spaces of the other polarity of the other orientation. This often magnifies the best focus differences, and challenges the tip-to-tip and tip-to-line imaging.

Reduction of pupil fill
A consequence of SMO and shifting focus windows has been the reduction of pupil fill. In other words, the optimum illumination is necessarily an optimized overlap of the preferred illuminations for the various patterns that need to be considered. This leads to lower pupil fill providing better results. However, throughput is affected below 20% pupil fill due to absorption.

Phase shift masks
A commonly touted advantage of EUV has been the relative ease of lithography, as indicated by the ratio of feature size to the wavelength multiplied by the numerical aperture, also known as the k1 ratio. An 18 nm metal linewidth has a k1 of 0.44 for 13.5 nm wavelength, 0.33 NA, for example. For the k1 approaching 0.5, some weak resolution enhancement including attenuated phase shift masks has been used as essential to production with the ArF laser wavelength (193 nm), whereas this resolution enhancement is not available for EUV. In particular, 3D mask effects including scattering at the absorber edges distort the desired phase profile. Also, the phase profile is effectively derived from the plane wave spectrum reflected from the multilayer through the absorber rather than the incident plane wave. Without absorbers, near-field distortion also occurs at an etched multilayer sidewall due to the oblique incidence illumination; some light traverses only a limited number of bilayers near the sidewall. Additionally, the different polarizations (TE and TM) have different phase shifts. Fundamentally, a chromeless phase shift mask enables pitch splitting by suppression of the zeroth diffracted order on the mask, but fabricating a high quality phase shift mask for EUV is certainly not a trivial task. One possible way to achieve this is through spatial filtering at the Fourier plane of the mask pattern. At Lawrence Berkeley National Lab, the light of the zeroth order is a centrally obscured system, and the +/-1 diffracted orders will be captured by the clear aperture, providing a functional equivalent to the chromeless phase shift mask while using a conventional binary amplitude mask.

EUV photoresist exposure: the role of electrons
EUV light generates photoelectrons upon absorption by matter. These photoelectrons in turn generate secondary electrons, which slow down before engaging in chemical reactions. At sufficient doses 40 eV electrons are known to penetrate 180 nm thick resist leading to development. At a dose of 160 μC/cm2, corresponding to 15 mJ/cm2 EUV dose assuming one electron/photon, 30 eV electrons removed 7 nm of PMMA resist after standard development. For a higher 30 eV dose of 380 μC/cm2, equivalent to 36 mJ/cm2 at one electron/photon, 10.4 nm of PMMA resist are removed. These indicate the distances the electrons can travel in resist, regardless of direction.
The degree of photoelectron emission from the layer underlying the EUV photoresist has been shown to affect the depth of focus. Unfortunately, hardmask layers tend to increase photoelectron emission, degrading the depth of focus. Electrons from defocused images in the resist can also affect the best focus image.
The randomness of the number of secondary electrons is itself a source of stochastic behavior in EUV resist images. The scale length of electron blur itself has a distribution. Intel demonstrated with a rigorous simulation that EUV-released electrons scatter distances larger than 15 nm in EUV resists.
The electron blur is also affected by total internal reflection from the top surface of the resist film.
A more realistic description of the electron blur uses the difference of two exponential functions.
In chemically amplified resists, acid blur can help to smoothen edge roughness, but low spatial-frequency roughness still remains, whereas in metal oxide resists, even high spatial-frequency roughness remains, since there is no acid blur smoothing. More blur can smooth the smaller scale roughness, but at the cost of reduced image contrast.

Effect of underlying layers
Secondary electrons from layers underneath the resist can affect the resist profile as well as pattern collapse. Hence, selection of such both the underlayer and the layer under that layer are important considerations for EUV lithography. Moreover, the electrons from defocused images can aggravate the stochastic nature of the image.

Contamination effects
Resist outgassing
Due to the high efficiency of absorption of EUV by photoresists, heating and outgassing become primary concerns. One well-known issue is contamination deposition on the resist from ambient or outgassed hydrocarbons, which results from EUV- or electron-driven reactions. Organic photoresists outgas hydrocarbons while metal oxide photoresists outgas water and oxygen and metal (in a hydrogen ambient); the last is uncleanable. The carbon contamination is known to affect multilayer reflectivity while the oxygen is particularly harmful for the ruthenium capping layers (relatively stable under EUV and hydrogen conditions) on the EUV multilayer optics.
EUV resist degradation increases with dose, as evidenced by outgassing of key components.
Several studies have shown that EUV resists are thinned as dose increases.
For metal oxide resists, IMEC had shown that counterion loss was significant during EUV exposure.

Tin redeposition
Atomic hydrogen in the tool chambers is used to clean tin and carbon which deposit on the EUV optical surfaces. Atomic hydrogen is produced by EUV light directly photoionizing H2:

hν + H2 → H+ + H + e−.
Electrons generated in the above reaction may also dissociate H2 to form atomic hydrogen:

e− + H2 → H+ + H + 2e−.
The reaction with tin in the light source (e.g., tin on an optical surface in the source) to form volatile SnH4 (stannane) that can be pumped out from the source proceeds via the reaction

Sn(s) + 4 H(g) → SnH4(g).
The SnH4 can reach the coatings of other EUV optical surfaces, where it redeposits Sn via the reaction

SnH4 → Sn(s) + 2 H2(g).
Redeposition may also occur by other intermediate reactions.
The redeposited Sn might be subsequently removed by atomic-hydrogen exposure. However, overall, the tin cleaning efficiency (the ratio of the removed tin flux from a tin sample to the atomic-hydrogen flux to the tin sample) is less than 0.01%, due to both redeposition and hydrogen desorption, leading to formation of hydrogen molecules at the expense of atomic hydrogen. The tin cleaning efficiency for tin oxide is found roughly twice higher than that of tin (with a native oxide layer of ~2 nm on it). Injecting a small amount of oxygen to the light source may improve the tin cleaning rate.
Removal of tin (Sn) particles is crucial for maintaining mask performance, as Sn is used to generate EUV light and continuously contaminates EUV masks during lithography.

Hydrogen blistering
Hydrogen also reacts with metal-containing compounds to reduce them to metal, and diffuses through the silicon and molybdenum in the multilayer, eventually causing blistering. Capping layers that mitigate hydrogen-related damage often reduce reflectivity to well below 70%. Capping layers are known to be permeable to ambient gases including oxygen and hydrogen, as well as susceptible to the hydrogen-induced blistering defects. Hydrogen may also react with the capping layer, resulting in its removal. TSMC proposed some means for mitigating hydrogen blistering defects on EUV masks, which may impact productivity.

Tin spitting
Hydrogen can penetrate molten tin (Sn), creating hydrogen bubbles inside it. If the bubbles move at the molten tin surface, then it bursts with tin, resulting in tin spreading over a large angle range. This phenomenon is called tin spitting and is one of EUV Collector contamination sources.

Resist erosion
Hydrogen also reacts with resists to etch or decompose them. Besides photoresist, hydrogen plasmas can also etch silicon, albeit very slowly.

Membrane
To help mitigate the above effects, the latest EUV tool introduced in 2017, the NXE:3400B, features a membrane that separates the wafer from the projection optics of the tool, protecting the latter from outgassing from the resist on the wafer. The membrane contains layers which absorb DUV and IR radiation, and transmits 85–90% of the incident EUV radiation. There is of course, accumulated contamination from wafer outgassing as well as particles in general (although the latter are out of focus, they may still obstruct light).

EUV-induced plasma
EUV lithographic systems using EUV light operate in 1–10 Pa hydrogen background gas. The plasma is a source of VUV radiation as well as electrons and hydrogen ions This plasma is known to etch exposed materials.
In 2023, a study supported at TSMC was published which indicated net charging by electrons from the plasma as well as from electron emission. The charging was found to occur even outside the EUV exposure area, indicating that the surrounding area had been exposed to electrons.
Due to chemical sputtering of carbon by the hydrogen plasma, there can be generation of nanoparticles, which can obstruct the EUV resist exposure.

Mask defects
Reducing defects on extreme ultraviolet (EUV) masks is currently one of the most critical issues to be addressed for commercialization of EUV lithography. Defects can be buried underneath or within the multilayer stack or be on top of the multilayer stack. Mesas or protrusions form on the sputtering targets used for multilayer deposition, which may fall off as particles during the multilayer deposition. In fact, defects of atomic scale height (0.3–0.5 nm) with 100 nm FWHM can still be printable by exhibiting 10% CD impact. IBM and Toppan reported at Photomask Japan 2015 that smaller defects, e.g., 50 nm size, can have 10% CD impact even with 0.6 nm height, yet remain undetectable.
Furthermore, the edge of a phase defect will further reduce reflectivity by more than 10% if its deviation from flatness exceeds 3 degrees, due to the deviation from the target angle of incidence of 84 degrees with respect to the surface. Even if the defect height is shallow, the edge still deforms the overlying multilayer, producing an extended region where the multilayer is sloped. The more abrupt the deformation, the narrower the defect edge extension, the greater the loss in reflectivity.
EUV mask defect repair is also more complicated due to the across-slit illumination variation mentioned above. Due to the varying shadowing sensitivity across the slit, the repair deposition height must be controlled very carefully, being different at different positions across the EUV mask illumination slit.

Multilayer reflectivity random variations
GlobalFoundries and Lawrence Berkeley Labs carried out a Monte Carlo study to simulate the effects of intermixing between the molybdenum (Mo) and silicon (Si) layers in the multilayer that is used to reflect EUV light from the EUV mask. The results indicated high sensitivity to the atomic-scale variations of layer thickness. Such variations could not be detected by wide-area reflectivity measurements but would be significant on the scale of the critical dimension (CD). The local variation of reflectivity could be on the order of 10% for a few nm standard deviation.

Multilayer damage
Multiple EUV pulses at less than 10 mJ/cm2 could accumulate damage to a Ru-capped Mo/Si multilayer mirror optic element. The angle of incidence was 16° or 0.28 rads, which is within the range of angles for a 0.33 NA optical system.

Pellicles
Production EUV tools need a pellicle to protect the mask from contamination. Pellicles are normally expected to protect the mask from particles during transport, entry into or exit from the exposure chamber, as well as the exposure itself. Without pellicles, particle adders would reduce yield, which has not been an issue for conventional optical lithography with 193 nm light and pellicles. However, for EUV, the feasibility of pellicle use is severely challenged, due to the required thinness of the shielding films to prevent excessive EUV absorption. Particle contamination would be prohibitive if pellicles were not stable above 200 W, i.e., the targeted power for manufacturing.
Heating of the EUV mask pellicle (film temperature up to 750 K for 80 W incident power) is a significant concern, due to the resulting deformation and transmission decrease. ASML developed a 70 nm thick polysilicon pellicle membrane, which allows EUV transmission of 82%; however, less than half of the membranes survived expected EUV power levels. SiNx pellicle membranes also failed at 82 W equivalent EUV source power levels. At target 250 W levels, the pellicle is expected to reach 686 degrees Celsius, well over the melting point of aluminum. Alternative materials need to allow sufficient transmission as well as maintain mechanical and thermal stability. However, graphite, graphene or other carbon nanomaterials (nanosheets, nanotubes) are damaged by EUV due to the release of electrons and also too easily etched in the hydrogen cleaning plasma expected to be deployed in EUV scanners. Hydrogen plasmas can also etch silicon as well. A coating helps improve hydrogen resistance, but this reduces transmission and/or emissivity, and may also affect mechanical stability (e.g., bulging).
Wrinkles on pellicles can cause CD nonuniformity due to uneven absorption; this is worse for smaller wrinkles and more coherent illumination, i.e., lower pupil fill.
In the absence of pellicles, EUV mask cleanliness would have to be checked before actual product wafers are exposed, using wafers specially prepared for defect inspection. These wafers are inspected after printing for repeating defects indicating a dirty mask; if any are found, the mask must be cleaned and another set of inspection wafers are exposed, repeating the flow until the mask is clean. Any affected product wafers must be reworked.
TSMC reported starting limited use of its own pellicle in 2019 and continuing to expand afterwards, and Samsung is planning pellicle introduction in 2022. However, follow-up reporting indicating no users of EUV pellicles due to accelerated damage under higher power.
Without pellicles, yield can be significantly reduced by particles added onto the die exposure area of the mask. The particle size also determines if it can be fatal to yield.

Hydrogen bulging defects
As discussed above, with regard to contamination removal, hydrogen used in recent EUV systems can penetrate into the EUV mask layers. TSMC indicated in its patent that hydrogen would enter from the mask edge. Once trapped, bulge defects or blisters were produced, which could lead to film peeling. These are essentially the blister defects which arise after a sufficient number of EUV mask exposures in the hydrogen environment. TSMC proposed some means for mitigating hydrogen blistering defects on EUV masks, which may impact productivity.

EUV stochastic issues
EUV lithography is particularly sensitive to stochastic effects. In a large population of features printed by EUV, although the overwhelming majority are resolved, some suffer complete failure to print, e.g. missing holes or bridging lines. A known significant contribution to this effect is the dose used to print. This is related to shot noise, to be discussed further below. Due to the stochastic variations in arriving photon numbers, some areas designated to print actually fail to reach the threshold to print, leaving unexposed defect regions. Some areas may be overexposed, leading to excessive resist loss or crosslinking. The probability of stochastic failure increases exponentially as feature size decreases, and for the same feature size, increasing distance between features also significantly increases the probability.  Line cuts which are misshapen are a significant issue due to potential arcing and shorting. Yield requires detection of stochastic failures down to below 1e-12.
The tendency to stochastic defects is worse from defocus over a large pupil fill.

Multiple failure modes may exist for the same population. For example, besides bridging of trenches, the lines separating the trenches may be broken. This can be attributed to stochastic resist loss, from secondary electrons. The randomness of the number of secondary electrons is itself a source of stochastic behavior in EUV resist images.
The coexistence of stochastically underexposed and overexposed defect regions leads to a loss of dose window at a certain post-etch defect level between the low-dose and high-dose patterning cliffs. Hence, the resolution benefit from shorter wavelength is lost.
The resist underlayer also plays an important role. This could be due to the secondary electrons generated by the underlayer. Secondary electrons may remove over 10 nm of resist from the exposed edge.
Photon shot noise also leads to stochastic edge placement error. The photon shot noise can be compensated to some degree by blurring factors such as secondary electrons or acids in chemically amplified resists; when significant the blur also reduces the image contrast at the edge. An edge placement error (EPE) as large as 8.8 nm was measured for a 48 nm pitch EUV-printed metal pattern.
With the natural Poisson distribution due to the random arrival and absorption times of the photons, there is an expected natural dose (photon number) variation of at least several percent 3 sigma, making the exposure process susceptible to stochastic variations. The dose variation leads to a variation of the feature edge position, effectively becoming a blur component. Unlike the hard resolution limit imposed by diffraction, shot noise imposes a softer limit, with the main guideline being the ITRS line width roughness (LWR) spec of 8% (3s) of linewidth. Increasing the dose will reduce the shot noise, but this also requires higher source power.
The two issues of shot noise and EUV-released electrons point out two constraining factors: 1) keeping dose high enough to reduce shot noise to tolerable levels, but also 2) avoiding too high a dose due to the increased contribution of EUV-released photoelectrons and secondary electrons to the resist exposure process, increasing the edge blur and thereby limiting the resolution. Aside from the resolution impact, higher dose also increases outgassing and limits throughput, and crosslinking occurs at very high dose levels. For chemically amplified resists, higher dose exposure also increases line edge roughness due to acid generator decomposition. Also, an upper limit to how much dose can be increased is imposed by resist loss. 
Due to resist thinning with increased dose, EUV stochastic defectivity limits will define a narrow CD or dose window. The thinner resist at higher incident dose reduces absorption, and hence, absorbed dose. The thinner resist is also more susceptible to nanovoids leading to resist breaks.
Even with higher absorption at the same dose, EUV has a larger shot noise concern than the ArF (193 nm) wavelength, mainly because it is applied to thinner resists. There is also an extra component noise from the secondary electron emission.
Due to stochastic considerations, the IRDS 2022 lithography roadmap now acknowledges increasing doses for smaller feature sizes. 
EUV resolution will likely be compromised by stochastic effects. Stochastic defect densities have exceeded 1/cm2, at 36 nm pitch; this is aggravated by electron blur. In 2024, an EUV resist exposure by ASML revealed a missing+bridging 32 nm pitch contact hole defect density floor >0.25/cm2 (177 defects per wafer), made worse with thinner resist. ASML indicated 30 nm pitch would not use direct exposure but double patterning. Intel did not use EUV for 30 nm pitch. Besides lower absorbed photon density, the impact of stochastic effects on EUV resolution is also tied to the smaller molecular size of EUV resists.
DRAM scaling will also become difficult at 10-11 nm design rules, due to EUV stochastics. The storage node patterns, which are arranged in a hexagonal array, are particularly sensitive due to their reliance on hexapole EUV illumination, which splits the image into three different sub-images, each with a third of the dose.
Larger features may unexpectedly suffer from stochastic fluctuations as well, due to local peaks and valleys in their aerial image.
IMEC's yield model updated in 2024 indicated that increased EUV use for 5nm node and beyond resulted in reduced yield, due to higher stochastic defectivity at tighter pitches. 
As of 2025, stochastic defect probabilities were on the order of ppm, and could also vary over an order of magnitude, leading to erratic yields.

Pupil fill ratio
For pitches less than half-wavelength divided by numerical aperture, dipole illumination is necessary. This illumination fills at most a leaf-shaped area at the edge of the pupil. However, due to 3D effects in the EUV  mask, smaller pitches require even smaller portions of this leaf shape. Below 20% of the pupil, the throughput and dose stability begin to suffer. Higher numerical aperture allows a higher pupil fill to be used for the same pitch, but depth of focus is significantly reduced.
A larger pupil fill is more susceptible to stochastic fluctuations from point to point in the pupil.

Use with multiple-patterning
EUV is anticipated to use double-patterning at around 34 nm pitch with 0.33 NA. This resolution is equivalent to '1Y' for DRAM. In 2020, ASML reported that 5 nm M0 layer (30 nm minimum pitch) required double-patterning.
In H2 2018, TSMC confirmed that its 5 nm EUV scheme still used multi-patterning, also indicating that mask count did not decrease from its 7 nm node, which used extensive DUV multi-patterning, to its 5 nm node, which used extensive EUV. EDA vendors also indicated the continued use of multi-patterning flows. While Samsung introduced its own 7 nm process with EUV single-patterning, it encountered severe photon shot noise causing excessive line roughness, which required higher dose, resulting in lower throughput. TSMC's 5 nm node uses even tighter design rules. Samsung indicated smaller dimensions would have more severe shot noise.

In Intel's complementary lithography scheme at 20 nm half-pitch, EUV would be used only in a second line-cutting exposure after a first 193 nm line-printing exposure.
Multiple exposures would also be expected where two or more patterns in the same layer, e.g., different pitches or widths, must use different optimized source pupil shapes. For example, when considering a staggered bar array of 64 nm vertical pitch, changing the horizontal pitch from 64 nm to 90 nm changes the optimized illumination significantly. Source-mask optimization that is based on line-space gratings and tip-to-tip gratings only does not entail improvements for all parts of a logic pattern, e.g., a dense trench with a gap on one side.
In 2020, ASML reported that for the 3 nm node, center-to-center contact/via spacings of 40 nm or less would require double- or triple-patterning for some contact/via arrangements.
For the 24–36 nm metal pitch, it was found that using EUV as a (second) cutting exposure had a significantly wider process window than as a complete single exposure for the metal layer. However, using a second exposure in the LELE approach for double patterning does not get around the vulnerability to stochastic defects.
Multiple exposures of the same mask are also expected for defect management without pellicles, limiting productivity similarly to multiple-patterning.
Self-aligned litho-etch-litho-etch (SALELE) is a hybrid SADP/LELE technique whose implementation has started in 7 nm. 
Self-aligned litho-etch-litho-etch (SALELE) has become an accepted form of double-patterning to be used with EUV.
In order to avoid higher doses for alleviating stochastic effects (even for 36 nm vias) splitting the pattern, leading to double patterning or multipatterning, would lead to a better image quality. In fact, this occurs at large enough design rules (i.e., 36 nm) that it overlaps with DUV double patterning.

Single-patterning extension: anamorphic high-NA
A return to extended generations of single-patterning would be possible with higher numerical aperture (NA) tools. An NA of 0.45 could require retuning of a few percent. Increasing demagnification could avoid this retuning, but the reduced field size severely affects large patterns (one die per 26 mm × 33 mm field) such as the many-core multi-billion transistor 14 nm Xeon chips. by requiring field stitching of two mask exposures.
In 2015, ASML disclosed details of its anamorphic next-generation EUV scanner, with an NA of 0.55. These machines cost around USD 360 million. The demagnification is increased from 4× to 8× only in one direction (in the plane of incidence). However, the 0.55 NA has a much smaller depth of focus than immersion lithography. Also, an anamorphic 0.52 NA tool has been found to exhibit too much CD and placement variability for 5 nm node single exposure and multi-patterning cutting.
Depth of focus being reduced by increasing NA is also a concern, especially in comparison with multi-patterning exposures using 193 nm immersion lithography:

High-NA EUV tools focus horizontal and vertical lines differently from low-NA systems, due to the different demagnification for horizontal lines.
High-NA EUV tools also suffer from obscuration, which can cause errors in the imaging of certain patterns. 
Intel completed assembly of its first commercial high-NA tool, TWINSCAN EXE:5000, in its D1X R&D fab in Oregon in April 2024, with a second machine installed in October 2024. In late 2025, Intel installed the new TWINSCAN EXE:5200B for its upcoming 14A node, with volume production expected in 2027.
For sub-2nm nodes, high-NA EUV systems will be affected by a host of issues: throughput, new masks, polarization, thinner resists, and secondary electron blur and randomness. Reduced depth of focus requires resist thickness less than 30 nm, which in turn increases stochastic effects, due to reduced photon absorption.
Electron blur is estimated to be at least ~2 nm, which is enough to thwart the benefit of High-NA EUV lithography.
Beyond high-NA, ASML in 2024 announced plans for the development of a hyper-NA EUV tool with an NA beyond 0.55, such as an NA of 0.75 or 0.85. These machines could cost USD 720 million each and are expected to be available in 2030. A problem with Hyper-NA is polarization of the EUV light causing a reduction in image contrast.

Beyond EUV wavelength
A much shorter wavelength (~6.7 nm) would be beyond EUV, and is often referred to as BEUV (beyond extreme ultraviolet). With current technology, BEUV wavelengths would have worse shot noise effects without ensuring sufficient dose. (The generally accepted 'border' of UV is 10nm below which the (soft) X-ray region begins.)

## Related Connections

- Mentions:: [[ASML_Holding]]
- Manufactures:: [[Integrated_circuit]]
- Sustains:: [[Moores_law]]
- Is a:: [[Photolithography]]
- Uses:: [[Silicon]]
- Sustains:: [[Transistor]]
