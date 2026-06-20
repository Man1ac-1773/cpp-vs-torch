/* Visualisation framework using graphviz.

 * Call below function, with the final output node, and filename in the syntax given
 * build_dot(loss.node, "graph.dot");
 * Then compile the plaintext file into an actual png graph using dot command.
 * system("dot -Tpng graph.dot -o graph.png");
 * Pre-requisite of dot command is installation of package graphviz package on linux
 * sudo apt install graphviz
 */

#pragma once
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "value.h"

/* Build the simple text representation of graph;
 * output is stored in <filename>;
 * make sure extension of file is .dot
 */
inline void build_dot(const shared_ptr<Node>& root, const string& filename)
{
    set<shared_ptr<Node>> nodes;
    vector<shared_ptr<Node>> stack{root};

    // traverse graph
    while (!stack.empty())
    {
        auto v = stack.back();
        stack.pop_back();

        if (nodes.insert(v).second) // insert returns pair<iterator,bool>
        {
            for (const auto& prev : v->_prev)
            {
                stack.push_back(prev);
            }
        }
    }

    ofstream out(filename);
    if (!out.is_open())
        return;

    out << "digraph G {\n";
    out << "  rankdir=\"LR\";\n";
    out << "  node [shape=record];\n";

    for (const auto& n : nodes)
    {
        string uid = "\"" + to_string(n->id) + "\"";

        // value node
        out << "  " << uid << " [label=\"{ " << n->label << " | data=" << n->data << " | grad=" << n->grad
            << " }\"];\n";

        if (!n->op.empty())
        {
            string op_uid = "\"op" + to_string(n->id) + "\"";

            // operation node
            out << "  " << op_uid << " [label=\"" << n->op << "\", shape=circle];\n";

            // op -> value
            out << "  " << op_uid << " -> " << uid << ";\n";

            for (const auto& prev : n->_prev)
            {
                string prev_uid = "\"" + to_string(prev->id) + "\"";

                // prev value -> op
                out << "  " << prev_uid << " -> " << op_uid << ";\n";
            }
        }
    }

    out << "}\n";
}
