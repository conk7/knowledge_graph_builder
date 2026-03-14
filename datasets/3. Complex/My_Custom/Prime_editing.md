# Prime editing

Prime editing is a 'search-and-replace' genome editing technology in molecular biology by which the genome of living organisms may be modified. The technology directly writes new genetic information into a targeted DNA site. It uses a fusion protein, consisting of a catalytically impaired Cas9 endonuclease fused to an engineered reverse transcriptase enzyme, and a prime editing guide RNA (pegRNA), capable of identifying the target site and providing the new genetic information to replace the target DNA nucleotides. It mediates targeted insertions, deletions, and base-to-base conversions without the need for double strand breaks (DSBs) or donor DNA templates.
The technology has received mainstream press attention due to its potential uses in medical genetics. It utilizes methodologies similar to precursor genome editing technologies, including CRISPR/Cas9 and base editors. Prime editing has been used on some animal models of genetic disease and plants. In 2024, PM359, a gene therapy developed by Prime Medicine, became the first prime editor to enter clinical trials for human use. Prime Medicine reported in December 2025 that two chronic granulomatous disease patients treated with PM359 had been "effectively cured" of the disease.

Prime editing is a 'search-and-replace' genome editing technology in molecular biology by which the genome of living organisms may be modified. The technology directly writes new genetic information into a targeted DNA site. It uses a fusion protein, consisting of a catalytically impaired Cas9 endonuclease fused to an engineered reverse transcriptase enzyme, and a prime editing guide RNA (pegRNA), capable of identifying the target site and providing the new genetic information to replace the target DNA nucleotides. It mediates targeted insertions, deletions, and base-to-base conversions without the need for double strand breaks (DSBs) or donor DNA templates.
The technology has received mainstream press attention due to its potential uses in medical genetics. It utilizes methodologies similar to precursor genome editing technologies, including CRISPR/Cas9 and base editors. Prime editing has been used on some animal models of genetic disease and plants. In 2024, PM359, a gene therapy developed by Prime Medicine, became the first prime editor to enter clinical trials for human use. Prime Medicine reported in December 2025 that two chronic granulomatous disease patients treated with PM359 had been "effectively cured" of the disease.

Genome editing
Components
Prime editing involves three major components:

A prime editing guide RNA (pegRNA), capable of (i) identifying the target nucleotide sequence to be edited, and (ii) encoding new genetic information that replaces the targeted sequence. The pegRNA consists of an extended single guide RNA (sgRNA) containing a primer binding site (PBS) and a reverse transcriptase (RT) template sequence. During genome editing, the primer binding site allows the 3’ end of the nicked DNA strand to hybridize to the pegRNA, while the RT template serves as a template for the synthesis of edited genetic information.
A fusion protein consisting of a Cas9 H840A nickase fused to a Moloney Murine Leukemia Virus (M-MLV) reverse transcriptase.
Cas9 H840A nickase: the Cas9 enzyme contains two nuclease domains that can cleave DNA sequences, a RuvC domain that cleaves the non-target strand and a HNH domain that cleaves the target strand. The introduction of a H840A substitution in Cas9, through which the 840th amino acid histidine is replaced by an alanine, inactivates the HNH domain. With only the RuvC functioning domain, the catalytically impaired Cas9 introduces a single strand nick, hence the name nickase.
M-MLV reverse transcriptase: an enzyme that synthesizes DNA from a single-stranded RNA template.
A single guide RNA (sgRNA) that directs the Cas9 H840A nickase portion of the fusion protein to nick the non-edited DNA strand.

Mechanism
Genomic editing takes place by transfecting cells with the pegRNA and the fusion protein. Transfection is often accomplished by introducing vectors into a cell. Once internalized, the fusion protein nicks the target DNA sequence, exposing a 3’-hydroxyl group that can be used to initiate (prime) the reverse transcription of the RT template portion of the pegRNA. This results in a branched intermediate that contains two DNA flaps: a 3’ flap that contains the newly synthesized (edited) sequence, and a 5’ flap that contains the dispensable, unedited DNA sequence. The 5’ flap is then cleaved by structure-specific endonucleases or 5’ exonucleases. This process allows 3’ flap ligation, and creates a heteroduplex DNA composed of one edited strand and one unedited strand. The reannealed double stranded DNA contains nucleotide mismatches at the location where editing took place. In order to correct the mismatches, the cells exploit the intrinsic mismatch repair  (MMR)  mechanism, with two possible outcomes: (i) the information in the edited strand is copied into the complementary strand, permanently installing the edit; (ii) the original nucleotides are re-incorporated into the edited strand, excluding the edit.

Development process
During the development of this technology, several modifications were done to the components, in order to increase its effectiveness.

Prime editor 1
In the first system, a wild-type Moloney Murine Leukemia Virus (M-MLV) reverse transcriptase was fused to the Cas9 H840A nickase C-terminus. Detectable editing efficiencies were observed.

Prime editor 2
In order to enhance DNA-RNA affinity, enzyme processivity, and thermostability, five amino acid substitutions were incorporated into the M-MLV reverse transcriptase. The mutant M-MLV RT was then incorporated into PE1 to give rise to (Cas9 (H840A)-M-MLV RT(D200N/L603W/T330P/T306K/W313F)). Efficiency improvement was observed over PE1.

Prime editor 3
Despite its increased efficacy, the edit inserted by PE2 might still be removed due to DNA mismatch repair of the edited strand. To avoid this problem during DNA heteroduplex resolution, an additional single guide RNA (sgRNA) is introduced. This sgRNA is designed to match the edited sequence introduced by the pegRNA, but not the original allele. It directs the Cas9 nickase portion of the fusion protein to nick the unedited strand at a nearby site, opposite to the original nick. Nicking the non-edited strand causes the cell's natural repair system to copy the information in the edited strand to the complementary strand, permanently installing the edit. However, there are drawbacks to this system as nicking the unaltered strand can lead to additional undesired indels.

Prime editor 4
Prime editor 4 utilizes the same machinery as PE2, but also includes a plasmid that encodes for dominant negative MMR protein MLH1. Dominant negative MLH1 is able to essentially knock out endogenous MLH1 by inhibition, thereby reducing cellular MMR response and increasing prime editing efficiency.

Prime editor 5
Prime editor 5 utilizes the same machinery as PE3, but also includes a plasmid that encodes for dominant negative MLH1. Like PE4, this allows for a knockdown of endogenous MMR response, increasing the efficiency of prime editing.

Nuclease Prime Editor
Nuclease Prime Editor uses Cas9 nuclease instead of Cas9(H840A) nickase. Unlike prime editor 3 (PE3) that requires dual-nick at both DNA strands to induce efficient prime editing, Nuclease Prime Editor requires only a single pegRNA since the single-gRNA already creates double-strand break instead of single-strand nick.

Twin prime editing
The "twin prime editing" (twinPE) mechanism reported in 2021 allows editing large sequences of DNA – sequences as large as genes – which addresses the method's key drawback. It uses a prime editor protein and two prime editing guide RNAs.

History
Prime editing was developed in the lab of David R. Liu at the Broad Institute and disclosed in Anzalone et al. (2019). Since then prime editing and the research that produced it have received widespread scientific acclaim, being called "revolutionary" and an important part of the future of editing.

Development of epegRNAs
Prime editing efficiency can be increased with the use of engineered pegRNAs (epegRNAs). One common issue with traditional pegRNAs is degradation of the 3' end, leading to decreased PE efficiency. epegRNAs have a structured RNA motif added to their 3' end to prevent degradation.

Implications
Although additional research is required to improve the efficiency of prime editing, the technology offers promising scientific improvements over other gene editing tools. The prime editing technology has the potential to correct the vast majority of pathogenic alleles that cause genetic diseases, as it can repair insertions, deletions, and nucleotide substitutions.

Advantages
The prime editing tool offers advantages over traditional gene editing technologies. CRISPR/Cas9 edits rely on double-strand breaks and non-homologous end joining (NHEJ) or homology-directed repair (HDR) to fix DNA breaks, while the prime editing system employs single-strand breaks and DNA mismatch repair. This is an important feature of this technology given that DNA repair mechanisms such as NHEJ and HDR, generate unwanted, random insertions or deletions (indels). These are byproducts that complicate the retrieval of cells carrying the correct edit. Prime editors do not frequently create these indel byproducts, suggesting that prime editors can be more precise than earlier tools.
The prime editing system introduces single-stranded DNA breaks, as with base editors, instead of the double-stranded DNA breaks observed in other editing tools, such as CRISPR/Cas9 editing. Collectively, base editing and prime editing offer complementary strengths and weaknesses for making targeted transition mutations. Base editors offer higher editing efficiency and fewer indel byproducts if the desired edit is a transition point mutation and a PAM sequence exists roughly 15 bases from the target site. However, because the prime editing technology does not require a precisely positioned PAM sequence to target a nucleotide sequence, it offers more flexibility and editing precision. Remarkably, prime editors allow all types of substitutions, both transitions and transversions, to be installed into the target sequence. Cytosine base editing and adenine BE can already perform precise base transitions but base transversions cannot be achieved with these base editors. Prime editing performs transversions with high efficiency. PE can insert up to 44bp, delete up to 80, or combinations thereof.
Because the prime system involves three separate DNA binding events (between (i) the guide sequence and the target DNA, (ii) the primer binding site and the target DNA, and (iii) the 3’ end of the nicked DNA strand and the pegRNA), it has been suggested to have fewer undesirable off-target effects than CRISPR/Cas9.

Limitations
There is considerable interest in applying gene-editing methods to the treatment of diseases with a genetic component. However, there are multiple challenges associated with this approach. An effective treatment would require editing of a large number of target cells, which in turn would require an effective method of delivery and a great level of tissue specificity.
As of 2019, prime editing looks promising for relatively small genetic alterations, but more research needs to be conducted to evaluate whether the technology is efficient in making larger alterations, such as targeted insertions and deletions. Larger genetic alterations would require a longer RT template, which could hinder the efficient delivery of pegRNA to target cells. Furthermore, a pegRNA containing a long RT template could become vulnerable to damage caused by cellular enzymes. Prime editing in plants suffers from low efficiency ranging from zero to a few percent and needs significant improvement.
Some of these limitations have been mitigated by recent improvements to the prime editors, including motifs that protect pegRNAs from degradation. Further research is needed before prime editing could be used to correct pathogenic alleles in humans. Research has also shown that inhibition of certain MMR proteins, including MLH1 can improve prime editing efficiency.

Delivery method
Base editors used for prime editing require delivery of both a protein and RNA molecule into living cells. Introducing exogenous gene editing technologies into living organisms is a significant challenge. One potential way to introduce a base editor into animals and plants is to package the base editor into a viral capsid. The target organism can then be transduced by the virus to synthesize the base editor in vivo. Common laboratory vectors of transduction such as lentivirus cause immune responses in humans, so proposed human therapies often centered around adeno-associated virus (AAV) because AAV infections are largely asymptomatic. Unfortunately, the effective packaging capacity of AAV vectors is small, approximately 4.4kb not including inverted terminal repeats. As a comparison, an SpCas9-reverse transcriptase fusion protein is 6.3kb, which does not even account for the lengthened guide RNA necessary for targeting and priming the site of interest. However, successful delivery in mice has been achieved by splitting the editor into two AAV vectors or by using an adenovirus, which has a larger packaging capacity.

Applications
Prime editors may be used in gene drives. A prime editor may be incorporated into the Cleaver half of a Cleave and Rescue/ClvR system. In this case it is not meant to perform a precise alteration but instead to merely disrupt.
PE is among recently introduced technologies which allow the transfer of single-nucleotide polymorphisms (SNPs) from one individual crop plant to another. PE is precise enough to be used to recreate an arbitrary SNP in an arbitrary target, including deletions, insertions, and all 12 point mutations without also needing to perform a double-stranded break or carry a donating template.

## Related Connections

- Uses:: [[Cas9]]
- Follows from:: [[CRISPR]]
- Mentions:: [[DNA]]
- Mentions:: [[Gene_drive]]
