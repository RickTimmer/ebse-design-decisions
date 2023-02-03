import re
import string

# def addClassifierDataLatex(data, classifiers, vectorizers):
#   # data += "\pgfplotstableread[row sep=\\\\,col sep=&]{{"
#   # data += "}}\data{name}\n".format(name=name)
#   print(classifiers)
#   print(vectorizers)
#   return data

# def writeLatexFiles(data, figures):
#   data_file = open("latex/data.latex", "w")
#   data_file.write(data)
#   data_file.close()

#   figures_file = open("latex/figures.latex", "w")
#   figures_file.write(figures)
#   figures_file.close

def evaluationsToLatex(evaluations, increase_step):
  # print(evaluations)

  data = ""
  figures = ""
  table = str.maketrans('', '', string.ascii_lowercase)

  classifier_evals = [y for x, y in evaluations.groupby(evaluations["classifier"])]
  for classifier_evel in classifier_evals:
    vectorizer_evals = [y for x, y in classifier_evel.groupby(classifier_evel["vectorizer"])]
    figures += """\\begin{center}
    \\textbf{""" + classifier_evel.iloc[0]["classifier"] + """}
\\end{center}\n"""
    figures += "\\begin{table}[H]\n"
    first = True
    for df in vectorizer_evals:
      classifier_name = df.iloc[0]["vectorizer"] + " " + df.iloc[0]["classifier"].replace(" ", "").translate(table)
      if not first:
        figures += "\\hfill"
      first = False
      df = df.rename(columns={"aimed size": "size"})
      df = df.drop("classifier", axis=1)
      df = df.drop("vectorizer", axis=1)
      df = df.drop("training size", axis=1)
      figures += "\\parbox{.45\\linewidth}{\n\\centering\n"
      figures += df.round(3).to_latex(index=False)
      figures += "\\caption{All iteration results for " + classifier_name + "}\n"
      figures += "\\label{tab:" + classifier_name + "}\n"
      figures += "}\n"
      
    figures += "\end{table}\n\n"

  max = evaluations["aimed size"].max()

  full_size_only = evaluations.loc[evaluations["aimed size"] == max]
  data += "\pgfplotstableread[row sep=\\\\,col sep=&]{\n"
  data += "    Classifier & Precision & Recall & F1-score \\\\\n"
  classifier_names = []
  for _, row in full_size_only.iterrows():
    classifier_name = row["vectorizer"] + " " + row["classifier"].replace(" ", "").translate(table)
    classifier_names.append(classifier_name)
    data += "    {classifier} & {precision} & {recall} & {f1} \\\\\n".format(
      classifier=classifier_name,
      precision=row["precision"],
      recall=row["recall"],
      f1=row["f1"]
    )
  data += "}\classifierdata\n\n"

  figures += """\\begin{{figure}}[H]
    \centering
    \\begin{{tikzpicture}}
        \\begin{{axis}}[
                ybar,
                width=1\\textwidth,
                height=.5\\textwidth,
                legend style={{at={{(0.5,1)}},
                anchor=north,legend columns=-1}},
                symbolic x coords={{{classifier_names}}},
                xticklabel style={{text width=1cm,align=center}},
                xtick=data,
            ]
            \\addplot table[x=Classifier,y=Precision]{{\classifierdata}};
            \\addplot table[x=Classifier,y=Recall]{{\classifierdata}};
            \\addplot table[x=Classifier,y=F1-score]{{\classifierdata}};
            \legend{{Precision, Recall, F1-score}}
        \end{{axis}}
    \end{{tikzpicture}}
    \caption{{Metrics per classifier for all labelled emails}}
    \label{{fig:classifier_metrics}}
\end{{figure}}\n\n""".format(classifier_names=",".join(classifier_names))

  currentRow = ""
  combinations = []
  for _, row in evaluations.iterrows():
    name = row["classifier"].replace(" ", "") + row["vectorizer"]
    if name not in combinations:
      combinations.append(name)
    if currentRow != name:
      if currentRow != "":
        data += "}}\iterdata{name}\n\n".format(name=currentRow)
      data += "\pgfplotstableread[row sep=\\\\,col sep=&]{\n"
      data += "    Size & Precision & Recall & F1-score \\\\\n"
      currentRow = name
    
    data += "    {size} & {precision} & {recall} & {f1} \\\\\n".format(
      size=row["aimed size"],
      precision=row["precision"],
      recall=row["recall"],
      f1=row["f1"]
    )

  data += "}}\iterdata{name}\n\n".format(name=name)
  print(combinations)
  figure_sizes = "{{{sizes}}}".format(sizes=",".join(str(item) for item in [*range(increase_step, max, increase_step)]))
  for combination in combinations:
    figures += """\\begin{{figure}}[H]
    \centering
    \\begin{{tikzpicture}}
        \\begin{{axis}}[
                xlabel=Number of emails in the dataset,
                width=1\\textwidth,
                height=.7\\textwidth,
                legend style={{at={{(0.5,1)}},
                anchor=north,legend columns=-1}},
                xticklabels={sizes},
                x label style={{at={{(axis description cs:0.5,-0.07)}},anchor=north}},
                xticklabel style={{text width=1cm,align=center,rotate=90}},
                xtick=data,
                ymax=.6
            ]
            \\addplot table[x=Size,y=Precision]{{\iterdata{combination}}};
            \\addplot table[x=Size,y=Recall]{{\iterdata{combination}}};
            \\addplot table[x=Size,y=F1-score]{{\iterdata{combination}}};
            \legend{{Precision, Recall, F1-score}}
        \end{{axis}}
    \end{{tikzpicture}}
    \caption{{Metrics per iteration for {combination_fancy}}}
    \label{{fig:iteration_metrics_{combination}}}
\end{{figure}}\n\n""".format(sizes=figure_sizes, combination=combination, combination_fancy=re.sub(r"(\w)([A-Z])", r"\1 \2", combination))

  data_file = open("latex/data.latex", "w")
  data_file.write(data)
  data_file.close()
  figures_file = open("latex/figures.latex", "w")
  figures_file.write(figures)
  figures_file.close()

  print("Latex was written 🤮")

