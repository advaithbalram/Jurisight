from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-led-base-16384")  
model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-led-base-16384")

padding = "max_length" 

text="""Date of Interaction: November 8, 2024
Client Name: John M. Anderson
Matter: Breach of Contract in Construction Agreement
Attorney: Susan K. Meyers

Background and Introduction
This report summarizes the initial consultation with Mr. John M. Anderson, who sought legal advice regarding a dispute in a construction contract. The consultation took place on November 8, 2024, via an in-person meeting at our New York City office. The primary goal of this session was to understand Mr. Anderson’s objectives, review the relevant facts, and outline a preliminary strategy.

Client’s Objectives
Mr. Anderson's primary goal is to seek compensation for an alleged breach of contract by the contractor, ABC Builders Inc., who he claims did not fulfill the agreed-upon terms for constructing an addition to his residential property. Mr. Anderson emphasized the importance of a swift resolution, ideally through settlement, to avoid further delays in completing the construction. However, he is open to pursuing legal action if necessary to secure compensation for damages and additional costs.

Facts Presented by the Client
Mr. Anderson provided an account of events, beginning with the execution of the construction agreement with ABC Builders on June 15, 2024. Under this agreement, ABC Builders was to complete a two-room addition to Mr. Anderson’s property by October 1, 2024. The contract specified materials, labor costs, and construction milestones. However, Mr. Anderson states that ABC Builders halted work on September 20, citing unexpected cost increases and demanding an additional $20,000 above the agreed contract amount to complete the project. Mr. Anderson refused to pay the additional amount, asserting that it violated the fixed-price contract terms.

Relevant Documentation:
The client provided copies of the construction agreement, dated communications with ABC Builders, and an itemized list of additional expenses incurred due to the construction delay. These documents will require detailed examination to determine the obligations and rights of each party under the contract.

Legal Analysis
Based on the initial facts and documentation, it appears that ABC Builders may have breached the contract by unilaterally demanding additional payment and failing to complete the project within the agreed timeframe. Further analysis is needed to determine the enforceability of the fixed-price clause in the contract and evaluate whether Mr. Anderson could seek damages for project delays and additional expenses.

The key legal issues that need to be addressed are as follows:

Contractual Breach – Verification of the contract’s fixed-price terms and whether ABC Builders’ demand for additional funds constitutes a breach.
Remedies for Damages – Analysis of the potential recovery for Mr. Anderson, including compensatory damages for delay and additional living expenses if they are deemed necessary to complete the construction.
Next Steps
Document Review – A comprehensive review of the contract terms, along with all correspondence between Mr. Anderson and ABC Builders.
Legal Research – Examination of relevant state statutes and case law regarding construction contracts and the enforcement of fixed-price clauses.
Settlement Exploration – Given Mr. Anderson’s preference for a swift resolution, a formal demand letter will be prepared, outlining ABC Builders’ contractual breach and proposing settlement terms to complete the project.
Potential Litigation Preparation – If settlement discussions are unproductive, we will prepare for potential litigation to protect Mr. Anderson’s interests and recover damages.
Initial Recommendations
Mr. Anderson was advised that the next step will involve sending a formal demand letter to ABC Builders to attempt a resolution without court intervention. The letter will emphasize Mr. Anderson’s readiness to pursue legal remedies if the contractor fails to meet its obligations or provide adequate compensation. Mr. Anderson was also informed that litigation may be required if ABC Builders does not agree to a fair settlement, given the strength of his position under the fixed-price clause.

Conclusion
The consultation with Mr. Anderson provided a solid understanding of his concerns and goals. As we proceed, we will prioritize achieving a swift settlement, while preparing for litigation if necessary to protect his financial interests and ensure project completion. A follow-up meeting will be scheduled after the demand letter is sent to discuss ABC Builders’ response and potential next steps."""

input_tokenized = tokenizer.encode(text, return_tensors='pt',padding=padding,pad_to_max_length=True, max_length=6144,truncation=True)
summary_ids = model.generate(input_tokenized,
                                  num_beams=4,
                                  no_repeat_ngram_size=3,
                                  length_penalty=2,
                                  min_length=350,
                                  max_length=500)
summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]

print(summary)