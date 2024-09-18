// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "udf-sample.h"

#include <string>
#include <numeric>
#include <vector>
#include <cmath>
#include <sstream>
#include <iostream>

using namespace std;

// In this sample we are declaring a UDF that adds two ints and returns an int.
IMPALA_UDF_EXPORT
IntVal AddUdf(FunctionContext* context, const IntVal& arg1, const IntVal& arg2) {
  if (arg1.is_null || arg2.is_null) return IntVal::null();
  return IntVal(arg1.val + arg2.val);
}

// Multiple UDFs can be defined in the same file

// Classify input customer reviews.
IMPALA_UDF_EXPORT
StringVal ClassifyReviewsDefault(FunctionContext* context, const StringVal& input) {
  std::string request =
      std::string("Classify the following review as positive, neutral, or negative")
      + std::string(" and only include the uncapitalized category in the response: ")
      + std::string(reinterpret_cast<const char*>(input.ptr), input.len);
  StringVal prompt(request.c_str());
  return context->Functions()->ai_generate_text_default(context, prompt);
}

// Classify input customer reviews.
IMPALA_UDF_EXPORT
StringVal ClassifyReviews(FunctionContext* context, const StringVal& input) {
  std::string request =
      std::string("Classify the following review as positive, neutral, or negative")
      + std::string(" and only include the uncapitalized category in the response: ")
      + std::string(reinterpret_cast<const char*>(input.ptr), input.len);
  StringVal prompt(request.c_str());
  const StringVal endpoint("https://api.openai.com/v1/chat/completions");
  const StringVal model("gpt-3.5-turbo");
  const StringVal api_key_jceks_secret("open-ai-key");
  const StringVal params("{\"temperature\": 0.9, \"model\": \"gpt-4\"}");
  return context->Functions()->ai_generate_text(
      context, endpoint, prompt, model, api_key_jceks_secret, params);
}

static DoubleVal CosDistance(const vector<double>& a, const vector<double>& b) {
  double dot = inner_product(a.begin(), a.end(), b.begin(), 0.0);
  double dot_a = inner_product(a.begin(), a.end(), a.begin(), 0.0);
  double dot_b = inner_product(b.begin(), b.end(), b.begin(), 0.0);
  if (dot_b == 0 || dot_a == 0) {
    return DoubleVal::null();
  }
  // Compute cosine similarity
  double cos_similarity_a = dot / dot_a;
  double cos_similarity_b = dot / dot_b;
  double cos_similarity = sqrt(cos_similarity_a * cos_similarity_b);
  // Cosine distance is 1 - cosine similarity
  return DoubleVal(1.0 - cos_similarity);
}


static vector<double> ParseStringToDoubleVector(const StringVal& string_val) {
  
  vector<double> result(384);
  // vector looks like:
  // 0.0098765 0.0001340 0.00232322 0.00008787 .......]
  std::string input(reinterpret_cast<char*>(string_val.ptr), string_val.len);
  istringstream iss(input);
  string token;
  while (getline(iss, token, ' ')) {
    try {
      result.push_back(stod(token));
    } catch (const std::exception& e) {
      std::cout << "caught error " << e.what() << '\n';
      return result;
    }
  }
  return result;
}

static DoubleVal AiCosDistance(
  FunctionContext* ctx, const StringVal& column, const StringVal& input) {
  if (column.ptr == nullptr && column.len == 0) {
    return DoubleVal::null();
  }
  if (input.ptr == nullptr && input.len == 0) {
    return DoubleVal::null();
  }
  vector<double> col_parsed_data = ParseStringToDoubleVector(column);
  vector<double> input_parsed_data = ParseStringToDoubleVector(input);
  if (col_parsed_data.size() != input_parsed_data.size() || col_parsed_data.size() == 0) {
    return DoubleVal::null();
  }
  return CosDistance(col_parsed_data, input_parsed_data);
}


IMPALA_UDF_EXPORT
DoubleVal CosDistance(FunctionContext* context, const StringVal& input_a, const StringVal& input_b) {
  return AiCosDistance(context, input_a, input_b);
}

const char* ai_vector = "-0.024351057 0.016706724 0.037720565 -0.009163348 -0.030584529 -0.017057106 0.074209854 0.045743443 -0.009398543 0.009900006 -0.0057072034 0.0075812778 0.039579637 0.015210353 -0.08321807 0.01934417 -0.02198688 -0.03320649 -0.18101417 -0.1302361 -0.0022559583 0.013397509 -0.024293108 -0.036996894 0.0020279363 0.085679844 0.004727344 -0.0034175643 -0.0060347966 -0.11578715 0.066860214 -0.018657709 0.08784171 -0.007424338 -0.09356733 0.061401032 -0.08113723 0.012245809 0.0397398 -0.0025896602 -0.046631347 -0.08177892 0.0395029 0.015469557 0.043692924 0.10363917 -0.058459505 0.03673655 -0.05270954 0.040569205 -0.12588456 0.0064611426 -0.035835233 -0.010022924 -0.023881378 0.04593257 0.014544471 0.01948532 0.028413124 -0.05515347 0.024229588 -0.05303023 0.015251375 -0.004387147 0.09242431 0.033881035 -0.04736574 0.03205467 0.0013060704 -0.05115598 0.025863083 0.08150351 0.040890634 0.019212415 0.056598596 -0.05275788 0.030558988 -0.016659522 0.07881096 -0.054235213 -0.04220547 -0.045418225 -0.052757826 0.112251006 0.019931704 -0.042498983 -0.011669648 0.024266975 0.019151 -0.01658886 -0.010269772 -0.085343175 0.023876714 -0.042154524 -0.024961371 0.06195064 -0.0046208003 -0.15366066 0.0011165437 0.19419846 -0.033807274 0.026233405 -0.020399084 0.0013040346 -0.0010178227 -0.024138892 0.017546743 -0.009766972 0.07050209 -0.13769664 -0.11126458 -0.01720731 0.0660453 -0.051880494 0.0018888247 0.014571295 0.06078346 0.09623634 0.01354664 0.019316304 -0.00015062829 -0.026587037 -0.009333636 0.07073309 -0.0035104677 -0.062441908 -0.04467104 -8.746442e-34 -0.11184079 -0.042540323 0.02744119 0.06567865 0.003018117 -0.044158515 0.0052107275 -0.036837634 -0.015630445 0.020558145 -0.05914478 0.0073992666 -0.028742272 0.040498063 0.13394724 0.0068202405 -0.016448377 0.0821934 -0.022515614 -0.036441717 0.06527704 0.020908277 -0.0054790857 -0.03839641 0.0015072306 0.0074143456 0.016916836 -0.062762335 0.035325546 -0.014370697 0.027728975 0.08378643 -0.027798818 -0.0036009226 0.038998943 -0.02680731 -0.01868291 0.018948777 0.06531722 0.0071514603 0.0047367956 -0.0030088506 0.04015959 0.02791353 -0.0045505944 0.012210228 0.087140806 -0.0069900346 -0.03743484 0.011326348 0.015252931 0.013786065 0.017908763 -0.009938906 0.09020806 0.051722143 -0.03433678 0.004398592 -0.01887866 -0.0313104 0.0821778 0.016898485 -0.022148995 0.068330236 0.015708717 0.020298865 0.0062223496 0.016406342 0.12718828 0.014996997 -0.01084195 0.0018214001 0.031695127 -0.04428727 -0.05227616 0.0228571 0.05093637 -0.018934906 0.0027829593 -0.033688396 -0.13567978 -0.02709048 -0.035686895 -0.03356542 0.047813997 -0.0053832442 0.021343017 -0.040023733 0.01938627 0.011964052 -0.04331956 0.0004735672 0.03491946 0.018034616 -0.062457174 8.234621e-34 -0.09454362 0.013787707 -0.025486149 0.099019065 0.045540564 -0.020402037 -0.029611232 -0.059111003 0.042362798 0.08439158 -0.043285094 -0.0077547436 0.049321815 0.042063538 -0.036513668 0.014370359 0.0403215 -0.058897804 0.010082367 0.059860747 -0.027944114 0.034982517 -0.087722465 -0.060477126 -0.0048245476 0.08779645 -0.005482366 -0.021695096 -0.048163865 0.046967495 0.0084487125 -0.051706266 -0.020442555 0.08580606 -0.022581708 0.03437442 -0.01444614 0.003083687 -0.046450377 0.0304366 0.039748713 0.02960175 -0.09310783 0.05153836 0.007903679 -0.057064462 -0.041776605 0.08978434 -0.008228427 -0.04074617 -0.05346191 -0.034406148 -0.045298457 -0.09712179 -0.05818817 0.06092178 -0.009013768 0.006922297 0.012354314 0.062096868 -0.005945426 -0.08632387 0.05874474 0.053207185 -0.0535401 0.039480332 -0.04489803 0.07284847 -0.03956711 -0.051314183 0.10332079 0.02190178 0.00017842172 0.0094729215 0.022022363 -0.006830019 -0.12895834 -0.0098183155 -0.036411896 -0.042418078 0.004368101 -0.047644824 0.0065291408 0.10254067 -0.053263415 0.07334732 0.015861664 -0.029101042 0.025094822 -0.063110776 -0.043467432 0.067098536 0.014920063 -0.0009577197 -0.098717205 -1.468271e-08 0.004582468 -0.06706264 0.07648199 -0.019843876 0.0673919 0.0448267 -0.051026188 -0.0076847263 -0.029402915 0.028842036 0.018941687 -0.024192024 0.044078358 0.04416082 0.034458455 0.046520427 0.02163629 -0.0017773687 -0.0029977788 0.014338439 0.12526727 0.03432497 -0.014606536 0.039115537 -0.0023343624 -0.014367297 0.010071369 0.024427926 -0.041718293 0.088318996 -0.03151748 0.030088129 -0.0029309625 0.0048743538 0.09580857 0.09396298 0.014148704 -0.077243656 -0.03915723 -0.010616786 -0.008591076 0.06419292 -0.0330353 -0.030456418 0.09467977 -0.00900649 -0.029881915 -0.13298948 0.059831347 -0.011666699 0.007184108 0.035657465 0.004025756 0.056268476 0.07661441 -0.010082477 0.056753255 0.023477761 -0.0638443 0.08942691 0.04385241 0.043342337 0.04616466 -0.07034413";

IMPALA_UDF_EXPORT
BooleanVal AiTitles(FunctionContext* context, const StringVal& input_a) {
  StringVal input_b(ai_vector);
  DoubleVal distance = AiCosDistance(context, input_a, input_b);
  return BooleanVal(distance.val < 0.1);
}

IMPALA_UDF_EXPORT
DoubleVal AiScore(FunctionContext* context, const StringVal& input_a) {
  StringVal input_b(ai_vector);
  return AiCosDistance(context, input_a, input_b);
}
