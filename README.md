# offline-training-pipeline
This is the offline-training-pipeline for our project.

We adopt the offline training and online prediction Machine Learning System framework structure. 

We used the recent DistilBERT pre-trained large-scale NLP language model and fine-tuned it for the downstream fake news classification task.

Initial fine-tune training dataset are adopted from CONSTRAINT workshop of AAAI21. For offline routine training and updating in the future, we will adopt the Fakenewsnet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media. Fakenewsnet offers up-to-date datasets and is continuously been updated on a regular basis. We hope to track the lastest trend of popular fake news and broader fake news topic as well by doing offline-training of our model and achieve better performance in the online prediction.


References:

@misc{patwa2020fighting, 
  title={Fighting an Infodemic: COVID-19 Fake News Dataset}, 
  author={Parth Patwa and Shivam Sharma and Srinivas PYKL and Vineeth Guptha and Gitanjali Kumari and Md Shad Akhtar and Asif Ekbal and Amitava Das and Tanmoy Chakraborty}, 
  year={2020}, 
  eprint={2011.03327}, 
  archivePrefix={arXiv}, 
  primaryClass={cs.CL} 
}

@article{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}

@article{shu2020fakenewsnet,
  title={Fakenewsnet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media},
  author={Shu, Kai and Mahudeswaran, Deepak and Wang, Suhang and Lee, Dongwon and Liu, Huan},
  journal={Big data},
  volume={8},
  number={3},
  pages={171--188},
  year={2020},
  publisher={Mary Ann Liebert, Inc., publishers 140 Huguenot Street, 3rd Floor New~…}
}
