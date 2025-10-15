using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace BayesianChangePoint
{
    class Program
    {
        static void Main(string[] args)
        {
            /* Original data with missing values
            int[] disaster_data = new int[] {4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                                             3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                                             2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
                                             1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                                             0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                                             3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                                             0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1 };
            */                                 

            int[] disaster_data = {4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                                   3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                                   2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                                   1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                                   0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                                   3, 3, 1, 2, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                                   0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1 };

            Microsoft.ML.Probabilistic.Models.Range n = new Microsoft.ML.Probabilistic.Models.Range(disaster_data.Length).Named("years_idx");

            var switchpoint = Variable.DiscreteUniform(n.Clone()).Named("switchpoint");
            switchpoint.Name = nameof(switchpoint);

            var data = Variable.Array<int>(n).Named("data");

            using (Variable.Switch(switchpoint))
            {
                var early_rate = Variable.GammaFromShapeAndRate(2.0, 2.0).Named("early_rate");
                var late_rate = Variable.GammaFromShapeAndRate(2.0, 2.0).Named("late_rate");
                using (ForEachBlock block = Variable.ForEach(n))
                {
                    using (Variable.If(switchpoint > block.Index))
                        data[block.Index] = Variable.Poisson(early_rate);
                    using (Variable.IfNot(switchpoint > block.Index))
                        data[block.Index] = Variable.Poisson(late_rate);
                }
            }


            data.ObservedValue = disaster_data;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.GenerateInMemory = false;
            engine.Compiler.WriteSourceFiles = true;
            engine.Compiler.IncludeDebugInformation = true;

            var switchpointMarginal = engine.Infer<Discrete>(switchpoint);
            Console.WriteLine(switchpointMarginal);

            // Console.ReadKey();
        }
    }
}
